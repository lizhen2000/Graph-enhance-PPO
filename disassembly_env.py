import torch
from torch_geometric.data import Data
from problem_set import *
import math
from utils import calculate_carbonems, calculate_profit, calculate_time, DynamicRewardScaler


class DisassemblyGraph:
    def __init__(self, edge_file = 'cxxl80.txt'):
        #加载依赖关系文件
        self.edges = self.load_edges(edge_file)
        self.edge_index = self.build_edge_index()

        #初始化节点特征（静态）
        self.node_features = torch.tensor([[
            vp[0][i],   #拆卸时间
            vp[1][i],   #拆卸利润
            vp[4][i],   #拆卸方向
            vp[5][i],   #连接件拆卸工具
            vp[6][i],   #单元拆卸工具
            dytpf[i],   #拆卸碳排放
            cxcb[i]     #拆卸成本   
        ] for i  in range(PARTCOUNT + 1)], dtype=torch.float32).to(device)

        self.edge_index = self.edge_index.to(device)
    
    def load_edges(self, path):
        edges = []
        with open(path) as f:
            for line in f:
                src, dst, _ = map(int, line.strip().split())
                edges.append((src, dst))
        return edges
    
    def build_edge_index(self):

        edge_index = torch.tensor(
            [[e[0] for e in self.edges],
             [e[1] for e in self.edges]],
             dtype=torch.long
        )
        return edge_index
    
    def calculate_in_degree(self, valid_edges):

        in_degree = {i: 0 for i in range(PARTCOUNT + 1)}
        for src, dst in valid_edges:
            in_degree[dst] += 1
        return in_degree
    
    def get_current_graph(self, removed):

        valid_edges = []
        for src, dst in self.edges:
            if src not in removed and dst not in removed:
                valid_edges.append([src, dst])
        
        current_edge_index = torch.tensor(
            [[e[0] for e in valid_edges],
             [e[1] for e in valid_edges]],
             dtype=torch.long
        )

        in_degree = self.calculate_in_degree(valid_edges)

        mask = []
        for i in range(PARTCOUNT):
            if i in removed or in_degree[i] != 0:
                mask.append(False)
            else:
                mask.append(True)

        targets_removed = TARGET_PARTS.issubset(removed)
        mask.append(targets_removed)    

        dynamic_features = torch.cat([
            self.node_features,
            torch.tensor([[int(i in removed)] for i in range(PARTCOUNT + 1)], dtype=torch.float32, device=device),
            torch.tensor([[in_degree[i]] for i in range(PARTCOUNT + 1)], dtype = torch.float32, device=device)
        ], dim=1)
        

        return Data(x=dynamic_features,
                    edge_index=current_edge_index.to(device),
                    mask = torch.tensor(mask, device=device )
                    )

class DisassemblyEnv:
    def __init__(self, heuristic_pareto_file='pareto_frontier.json'):
        self.graph = DisassemblyGraph()  # 有向图，Data类型数据
        self.current_graph = None   # 动态更新的图
        self.removed = set()  # 已经拆卸的零部件定义为空集合
        self.normalizer = DynamicRewardScaler()  # 奖励归一化器
        self.current_preference = None  # 当前偏好向量
        self.epsilon = 1e-6

        heuristic_pareto = self.load_heuristic_pareto(heuristic_pareto_file)

        heuristic_pareto_norm = self.convert_and_norm_pareto(heuristic_pareto)

        self.reward_calculator = PreferenceGuidedReward(heuristic_pareto_norm)

        self.graph.x = self.graph.node_features.to(device)
        self.graph.edge_index = self.graph.edge_index.to(device)

    
    def load_heuristic_pareto(self, file_path):
        if not file_path:
            print("未指定启发式算法得到的Pareto前沿文件,使用空集")
            return []
        
        try:
            import json
            with open(file_path, 'r') as f:
                pareto_data = json.load(f)
            print(f"成功加载启发式算法得到的Pareto前沿，共{len(pareto_data)}个解")
            return pareto_data
        except Exception as e:
            print(f"加载启发式算法得到的Pareto前沿失败: {e}")
            return []
    
    def convert_and_norm_pareto(self, ParetoSet):
        
        converted_ParetoSet = []
        for item in ParetoSet:
            converted_ParetoSet.append([item['time'], item['profit'], item['carbon']])
        
        converted_and_norm_ParetoSet = []

        for i in converted_ParetoSet:
            time_transformed = 1.0 / (i[0] + self.epsilon)
            time_norm = self.normalizer.normalize(time_transformed, 'time')
            profit_norm = self.normalizer.normalize(i[1], 'profit')
            carbon_norm = self.normalizer.normalize(i[2], 'carbon')
            converted_and_norm_ParetoSet.append([time_norm, profit_norm, carbon_norm])
        
        return converted_and_norm_ParetoSet
        

    
    def reset(self, preference=None):
        self.removed = set()
        self.disassembled_sequence = []
        self.current_graph = self.graph.get_current_graph(self.removed)
        self.current_graph.x = self.current_graph.x.to(torch.float32).to(device)
        self.current_graph.edge_index = self.current_graph.edge_index.to(device)
        self.current_graph.mask = self.current_graph.mask.to(device)       
        self.current_preference = preference.to(device) if preference is not None else None
        return self.current_graph
    
    def step(self, action, preference):
        done = False
        is_stop = (action == STOP_ACTION)

        if is_stop:
            done = True
        else:
            self.removed.add(action)
            done = (len(self.removed) == PARTCOUNT)
        
        self.disassembled_sequence.append(action)
        
        #sequence = list(self.removed)
        #print(f"action: {action}")
        #print(f"sequence: {sequence}")
        reward = self.reward_calculator.calculate_reward(self.disassembled_sequence, done, preference)

        new_graph = self.graph.get_current_graph(self.removed)
        self.current_graph = new_graph

        return new_graph, reward, done
    


class PreferenceGuidedReward:
    def __init__(self, heuristic_pareto_frontier=None):

        self.normalizer = DynamicRewardScaler()
        self.heuristic_pareto = heuristic_pareto_frontier or []
        self.discovered_solutions = []
        self.epsilon = 1e-6

        self.solution_history = []
        self.solution_counts = {}
        self.max_history_size = 150
        
    def calculate_reward(self, sequence, done, preference):
 
        time_cost = calculate_time(sequence)
        profit = calculate_profit(sequence)
        carbon_emission = calculate_carbonems(sequence)
        

        #print(f"time:{time_cost}, profit:{profit}, carbon: {carbon_emission}")

        time_cost_transformed = 1.0 / (time_cost + self.epsilon)

        time_norm = self.normalizer.normalize(time_cost_transformed, 'time')
        profit_norm = self.normalizer.normalize(profit, 'profit')
        carbon_norm = self.normalizer.normalize(carbon_emission, 'carbon')
        #print(f"time_norm:{time_norm}, profit_norm:{profit_norm}, carbon_norm: {carbon_norm}")
        current_solution = torch.tensor(
            [time_norm, profit_norm, carbon_norm],
            dtype=torch.float32,
            device=device
        )
        
        current_score = (
            preference[0]*time_norm +
            preference[1]*profit_norm +
            preference[2]*carbon_norm
        )

        if not done:
            if len(sequence) == 1:
                self.last_score = 0.0

            step_reward = current_score - self.last_score
            self.last_score = current_score

            return step_reward

        #偏好投影奖励
        preference_tensor = preference.clone().detach().to(device)
        preference_norm = preference_tensor / (torch.norm(preference_tensor) + self.epsilon)
        preference_projection = torch.dot(current_solution, preference_norm)

        
        #Pareto提升奖励
        pareto_improvement_reward = self._calculate_pareto_improvement(current_solution, preference)
        
        #角度偏差惩罚
        solution_norm = current_solution / (torch.norm(current_solution) + self.epsilon)
        cosine_sim = torch.dot(preference_norm, solution_norm)
        angle = torch.acos(torch.clamp(cosine_sim, -1.0, 1.0))
        angle_penalty = self._calculate_angle_penalty(angle)

        #多样性惩罚
        solution_key = f"{time_cost:.1f}_{profit:.1f}_{carbon_emission:.1f}"
        repetition_penalty = 0.0
        
        if solution_key in self.solution_counts:
            count = self.solution_counts[solution_key]
            repetition_penalty = -min(50.0, count * 10.0)
            self.solution_counts[solution_key] += 1
        else:
            self.solution_counts[solution_key] = 1

        self.solution_history.append(solution_key)
        if len(self.solution_history) > self.max_history_size:
            old_key = self.solution_history.pop(0)
            self.solution_counts[old_key] -= 1
            if self.solution_counts[old_key] == 0:
                del self.solution_counts[old_key]

        base_reward = preference_projection * 50 + pareto_improvement_reward * 50
        
        final_reward = base_reward  + angle_penalty + repetition_penalty

        if not angle_penalty and pareto_improvement_reward > 0.7:
            self.discovered_solutions.append(current_solution)
        
        return final_reward


    def _dominates(self, sol_a, sol_b):
        return (
            all(sol_a[i] >= sol_b[i] for i in range(len(sol_a))) and 
            any(sol_a[i] > sol_b[i] for i in range(len(sol_a)))
        )
    
       
    def _calculate_pareto_improvement(self, current_solution, preference):
        combined_pareto = self.heuristic_pareto + [
            s.cpu().tolist() for s in self.discovered_solutions
        ]
    
        if not combined_pareto:
            return 0.8
    
        preference_tensor = preference.clone().detach()
        preference_norm = preference_tensor / (torch.norm(preference_tensor) + self.epsilon)
        preference_projection = torch.dot(current_solution, preference_norm)
    
        projections = []
        dominated_by_any = False
        dominates_any = False
    
        for solution in combined_pareto:
            solution_tensor = torch.tensor(solution, dtype=torch.float32, device=device)
        
            if self._dominates(solution_tensor, current_solution):
                dominated_by_any = True
            if self._dominates(current_solution, solution_tensor):
                dominates_any = True
            
            solution_projection = torch.dot(solution_tensor, preference_norm)
            projections.append(solution_projection.item())
    
        max_existing_projection = max(projections) if projections else 0
    
        projection_ratio = preference_projection.item() / (max_existing_projection + self.epsilon)
    
        if projection_ratio > 1.05:
            direction_reward = 1.0
        elif projection_ratio > 1.0:
            direction_reward = 0.9
        elif projection_ratio > 0.95:
            direction_reward = 0.8
        elif projection_ratio > 0.9:
            direction_reward = 0.7
        elif projection_ratio > 0.8:
            direction_reward = 0.5
        else:
            direction_reward = 0.3

        if not dominated_by_any and dominates_any:
            return min(1.0, direction_reward + 0.1)
        elif dominated_by_any:
            return max(0.3, direction_reward - 0.1)
        else:
            return direction_reward 
        
    def _calculate_angle_penalty(self, angle):
        if angle < math.pi/12:
            return 0
        elif angle <= math.pi/4:
            return -20 * ((angle - math.pi/12) / (math.pi/4 - math.pi/12))
        else:
            return -20 - 150.0 * ((angle - math.pi/4) / (math.pi/2 - math.pi/4))
    
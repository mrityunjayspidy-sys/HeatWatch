import random
import numpy as np
import pandas as pd
from heat_mitigation_engine import engine

# Cost Priors per Intervention Type per Grid Cell (Unit Cost in USD)
INTERVENTION_COSTS = {
    0: 0.0,       # None
    1: 1500.0,    # Tree Canopy Planting
    2: 800.0,     # Cool Reflective Roof Coating
    3: 3500.0,    # Water Retention Pond
    4: 1200.0     # Photovoltaic Shade Canopy
}

INTERVENTION_NAMES = {
    0: "None",
    1: "Trees & Canopy",
    2: "Cool Reflective Roof",
    3: "Water Body / Pond",
    4: "Solar Shade Canopy"
}

# Cooling effect priors (Delta Temp °C)
INTERVENTION_EFFECTS = {
    0: {"d_temp": 0.0,  "d_ndvi": 0.0,   "d_albedo": 0.0},
    1: {"d_temp": -2.8, "d_ndvi": 0.25,  "d_albedo": 0.08},
    2: {"d_temp": -2.1, "d_ndvi": 0.0,   "d_albedo": 0.35},
    3: {"d_temp": -3.5, "d_ndvi": 0.05,  "d_albedo": 0.05},
    4: {"d_temp": -1.6, "d_ndvi": 0.0,   "d_albedo": 0.15}
}

class GeneticAlgorithmCoolingOptimizer:
    def __init__(self, population_size=30, generations=25, mutation_rate=0.08, cost_budget=25000.0):
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.cost_budget = cost_budget

    def evaluate_chromosome(self, chromosome, grid_cells, baseline_lst_mean, equity_weighted=True):
        """
        Stage B Fitness Function:
        Fitness = Simulated Citywide Temperature Drop (°C) - Budget Penalty + Equity/Vulnerability Weighting
        """
        total_cost = 0.0
        total_temp_drop = 0.0
        weighted_temp_drop = 0.0
        total_vulnerability = 0.0
        n_cells = len(grid_cells)

        for gene, cell in zip(chromosome, grid_cells):
            cost = INTERVENTION_COSTS[gene]
            total_cost += cost
            effect = INTERVENTION_EFFECTS[gene]
            cell_drop = abs(effect['d_temp'])
            
            vuln_weight = cell.get('vulnerability_weight', 1.0) if equity_weighted else 1.0
            
            total_temp_drop += cell_drop
            weighted_temp_drop += cell_drop * vuln_weight
            total_vulnerability += vuln_weight

        mean_temp_drop = total_temp_drop / n_cells if n_cells > 0 else 0.0
        mean_weighted_drop = weighted_temp_drop / total_vulnerability if total_vulnerability > 0 else mean_temp_drop

        budget_penalty = 0.0
        if total_cost > self.cost_budget:
            budget_penalty = ((total_cost - self.cost_budget) / 1000.0) * 0.85

        fitness = mean_weighted_drop - budget_penalty
        return fitness, mean_temp_drop, total_cost

    def optimize(self, lat: float, lon: float, budget: float = 25000.0, num_grid_cells: int = 16):
        self.cost_budget = budget
        
        step = 0.007
        rows = int(np.sqrt(num_grid_cells))
        cols = rows

        grid_cells = []
        for r in range(rows):
            for c in range(cols):
                cell_lat = lat + (r - rows/2) * step
                cell_lon = lon + (c - cols/2) * step
                base_lst = engine.predict_baseline(cell_lat, cell_lon, 5, 0.08, 0.12, -0.15, 50.0)
                
                # Equity / Vulnerability Weight based on baseline heat stress
                vuln_weight = 1.0 + max(0.0, (base_lst - 32.0) / 10.0)

                grid_cells.append({
                    "id": f"cell_{r}_{c}",
                    "lat": cell_lat,
                    "lon": cell_lon,
                    "baseline_lst": base_lst,
                    "vulnerability_weight": vuln_weight
                })

        baseline_lst_mean = np.mean([c['baseline_lst'] for c in grid_cells])

        # Step 13: Initialize Population of Random Intervention Maps (respecting budget)
        population = []
        for _ in range(self.pop_size):
            chromosome = [random.randint(0, 4) for _ in range(num_grid_cells)]
            population.append(chromosome)

        best_chromosome = None
        best_fitness = -999.0
        best_metrics = (0.0, 0.0)

        # Step 13 Cont.: Evolutionary Loop (Selection -> Crossover -> Mutation)
        for gen in range(self.generations):
            evaluated_pop = []
            for chrom in population:
                fit, drop, cost = self.evaluate_chromosome(chrom, grid_cells, baseline_lst_mean)
                evaluated_pop.append((fit, drop, cost, chrom))

            # Selection (Tournament / Rank)
            evaluated_pop.sort(key=lambda x: x[0], reverse=True)

            if evaluated_pop[0][0] > best_fitness:
                best_fitness = evaluated_pop[0][0]
                best_metrics = (evaluated_pop[0][1], evaluated_pop[0][2])
                best_chromosome = evaluated_pop[0][3]

            # Elitism: Keep top 20%
            survivors_count = max(2, int(self.pop_size * 0.2))
            new_population = [x[3] for x in evaluated_pop[:survivors_count]]

            # Crossover & Mutation
            while len(new_population) < self.pop_size:
                p1 = random.choice(evaluated_pop[:10])[3]
                p2 = random.choice(evaluated_pop[:10])[3]
                
                cut = random.randint(1, num_grid_cells - 1)
                child = p1[:cut] + p2[cut:]

                for i in range(len(child)):
                    if random.random() < self.mutation_rate:
                        child[i] = random.randint(0, 4)

                new_population.append(child)

            population = new_population

        # Step 14: Output optimal intervention map + expected citywide temperature reduction + cost breakdown
        optimal_plan = []
        total_cost = 0.0
        for gene, cell in zip(best_chromosome, grid_cells):
            cost = INTERVENTION_COSTS[gene]
            total_cost += cost
            effect = INTERVENTION_EFFECTS[gene]
            opt_lst = cell['baseline_lst'] - abs(effect['d_temp'])
            optimal_plan.append({
                "cell_id": cell['id'],
                "lat": cell['lat'],
                "lon": cell['lon'],
                "intervention_id": gene,
                "intervention_name": INTERVENTION_NAMES[gene],
                "cost_usd": cost,
                "baseline_lst_celsius": round(cell['baseline_lst'], 1),
                "optimized_lst_celsius": round(opt_lst, 1),
                "equity_vulnerability_score": round(cell['vulnerability_weight'], 2)
            })

        mean_temp_drop = best_metrics[0]
        cost_effectiveness = round((mean_temp_drop / (total_cost / 1000.0)), 2) if total_cost > 0 else 0.0

        return {
            "status": "success",
            "algorithm": "Genetic Algorithm (GA) Optimization",
            "generations_run": self.generations,
            "population_size": self.pop_size,
            "budget_limit_usd": budget,
            "total_optimal_cost_usd": total_cost,
            "baseline_mean_lst_celsius": round(baseline_lst_mean, 2),
            "optimized_mean_lst_celsius": round(baseline_lst_mean - mean_temp_drop, 2),
            "citywide_temperature_reduction_celsius": round(mean_temp_drop, 2),
            "cost_effectiveness_celsius_per_k_usd": cost_effectiveness,
            "optimal_intervention_plan": optimal_plan
        }

ga_optimizer = GeneticAlgorithmCoolingOptimizer()

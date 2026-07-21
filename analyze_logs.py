import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    os.makedirs("analysis", exist_ok=True)
    print("Starting EDA on master logs...")
    
    # 1. Tournaments Master Analysis
    t_file = "logs/tournaments_master.csv"
    if os.path.exists(t_file):
        print("Analyzing tournaments...")
        df = pd.read_csv(t_file)
        
        # Survival rate by planner
        plt.figure(figsize=(10, 6))
        sns.barplot(x="planner", y="survival_rate", data=df, errorbar='sd', capsize=0.1)
        plt.title("Average Survival Rate by Planner")
        plt.ylabel("Survival Rate")
        plt.savefig("analysis/survival_rate.png", bbox_inches="tight")
        plt.close()

        # Efficiency by planner
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="planner", y="avg_efficiency", data=df)
        plt.title("Path Efficiency by Planner")
        plt.ylabel("Efficiency (Shortest Oracle / Actual Steps)")
        plt.savefig("analysis/efficiency_box.png", bbox_inches="tight")
        plt.close()
        
        # Mine death rate by planner
        plt.figure(figsize=(10, 6))
        sns.barplot(x="planner", y="mine_death_rate", data=df, errorbar='sd')
        plt.title("Mine Death Rate by Planner")
        plt.ylabel("Mine Death Rate")
        plt.savefig("analysis/mine_death_rate.png", bbox_inches="tight")
        plt.close()

    # 2. Experiments Master Analysis
    e_file = "logs/experiments_master.csv"
    if os.path.exists(e_file):
        print("Analyzing experiment events...")
        df_e = pd.read_csv(e_file)
        
        reached_cols = [c for c in df_e.columns if "reached" in c]
        if reached_cols:
            # Handle string boolean values if any
            def parse_bool(val):
                if isinstance(val, str):
                    return val.lower() == 'true'
                return bool(val)
                
            df_e['success_count'] = df_e[reached_cols].map(parse_bool).sum(axis=1)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x="config", y="success_count", data=df_e, errorbar='sd')
            plt.title("Robots Reaching Goal per Round (by Config)")
            plt.ylabel("Success Count (Out of 4)")
            plt.savefig("analysis/experiment_success_count.png", bbox_inches="tight")
            plt.close()

    print("EDA complete. Graphs generated in the analysis/ folder.")

if __name__ == "__main__":
    main()

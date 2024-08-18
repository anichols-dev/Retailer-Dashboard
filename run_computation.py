#%%
import subprocess

def install_requirements(bool):
    if bool:
        try:
            print("Installing required packages from requirements.txt...")
            subprocess.run(['pip', 'install', '-r', 'instructions/requirements.txt'], check=True)
            print("Successfully installed required packages. \n")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while installing packages: {e}")

def run_file(file_name):
    try:
        print(f'Running {file_name}....')
        subprocess.run(['python3', file_name], check=True)
        print(f"Successfully ran {file_name} \n")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {file_name}: {e}")
        return None
    return True

if __name__ == "__main__":
    install_requirements(False)
    
    files = ["Cluster_Retailers.py", "Populate_Adjacents.py", "Neighborhood_Adjacents.py", "Basket_Insights.py", "Market_Share.py"]

    for file in files:
        if not run_file(file):
            break
    
    print("Retailer Computation is Finished!")
# %%



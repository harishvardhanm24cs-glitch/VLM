import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_script(script_name):
    logging.info(f"Starting {script_name}...")
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode != 0:
            logging.error(f"{script_name} failed with exit code {process.returncode}")
            sys.exit(process.returncode)
            
        logging.info(f"{script_name} completed successfully.")
    except Exception as e:
        logging.error(f"Error running {script_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_script("d:/VLM/create_cropped_dataset.py")
    run_script("d:/VLM/finetune_cropped.py")
    logging.info("FULL PIPELINE FINISHED SUCCESSFULLY.")

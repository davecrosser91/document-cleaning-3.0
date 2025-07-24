#!/usr/bin/env python
"""
RunPod training helper for document cleaning autoencoder.
This script provides utilities to launch and manage training jobs on RunPod.
"""

import argparse
import json
import os
import re
import sys
import subprocess
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv

load_dotenv()

class RunpodTrainer:
    """Helper class to manage training on RunPod."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        pod_type: str = "GPU",
        gpu_type_id: str = "NVIDIA RTX A4000",  # Choose appropriate GPU
        template_id: str = "runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel",  # PyTorch template
        container_disk_size_gb: int = 10,
        volume_disk_size_gb: int = 20,
        volume_mount_path: str = "/workspace",
        name: Optional[str] = None,
    ):
        """Initialize RunPod trainer.
        
        Args:
            api_key: RunPod API key (defaults to RUNPOD_API_KEY env var)
            pod_type: Type of pod (GPU or CPU)
            gpu_type_id: GPU type to use
            template_id: Docker template ID
            container_disk_size_gb: Container disk size in GB
            volume_disk_size_gb: Volume disk size in GB
            volume_mount_path: Path to mount volume
            name: Name for the pod
        """
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "RunPod API key not provided. Set RUNPOD_API_KEY environment variable or pass api_key."
            )
        
        self.pod_type = pod_type
        self.gpu_type_id = gpu_type_id
        self.template_id = template_id
        self.container_disk_size_gb = container_disk_size_gb
        self.volume_disk_size_gb = volume_disk_size_gb
        self.volume_mount_path = volume_mount_path
        self.name = name or f"autoencoder-train-{int(time.time())}"
        self.pod_id = None
    
    def _run_command(self, cmd: List[str], expect_json: bool = True) -> Dict[str, Any]:
        """Run a command and return parsed JSON output.
        
        Args:
            cmd: Command to run
            expect_json: Whether to expect JSON output
            
        Returns:
            Parsed JSON output or dict with stdout
        """
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            if not expect_json:
                return {"stdout": result.stdout.strip()}
                
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                # Handle pod creation special case
                if "pod" in result.stdout and "created for" in result.stdout:
                    # Extract pod ID from output like: pod "yw9ptgjwjh3kon" created for $0.170 / hr
                    import re
                    pod_id_match = re.search(r'pod "([a-z0-9]+)" created', result.stdout)
                    if pod_id_match:
                        pod_id = pod_id_match.group(1)
                        return {"id": pod_id, "stdout": result.stdout.strip()}
                
                print(f"Error parsing JSON output: {result.stdout}")
                return {"stdout": result.stdout.strip()}
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            print(f"STDERR: {e.stderr}")
            sys.exit(1)
    
    def create_pod(self) -> str:
        """Create a RunPod pod.
        
        Returns:
            Pod ID
        """
        print(f"Creating RunPod pod: {self.name}")
        
        cmd = [
            "runpodctl",
            "create",
            "pod",
            "--gpuType", self.gpu_type_id,
            "--containerDiskSize", str(self.container_disk_size_gb),
            "--volumeSize", str(self.volume_disk_size_gb),
            "--volumePath", self.volume_mount_path,
            "--name", self.name,
            "--imageName", self.template_id,
            "--ports", "8888/http",  # For Jupyter if needed
        ]
        
        result = self._run_command(cmd)
        self.pod_id = result.get("id")
        
        if not self.pod_id:
            raise ValueError("Failed to create pod: no pod ID returned")
        
        print(f"Pod created with ID: {self.pod_id}")
        return self.pod_id
    
    def wait_for_pod_ready(self, timeout_seconds: int = 300) -> bool:
        """Wait for pod to be ready.
        
        Args:
            timeout_seconds: Timeout in seconds
            
        Returns:
            True if pod is ready, False otherwise
        """
        if not self.pod_id:
            raise ValueError("No pod ID available")
        
        print(f"Waiting for pod {self.pod_id} to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            cmd = ["runpodctl", "get", "pod", self.pod_id]
            result = self._run_command(cmd)
            
            status = result.get("status")
            if status == "RUNNING":
                print(f"Pod {self.pod_id} is ready!")
                return True
            
            print(f"Pod status: {status}. Waiting...")
            time.sleep(10)
        
        print(f"Timed out waiting for pod {self.pod_id} to be ready")
        return False
    
    def upload_code(self, local_path: Union[str, Path]) -> bool:
        """Upload essential code files to RunPod for training.
        
        Instead of uploading the entire directory, this method uploads only
        the essential files needed for training to avoid transfer issues.
        
        Args:
            local_path: Base local path (project root)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.pod_id:
            raise ValueError("No pod ID available")
        
        local_path = Path(local_path)
        if not local_path.exists():
            raise ValueError(f"Local path does not exist: {local_path}")
        
        print(f"Preparing to upload essential files from {local_path} to pod {self.pod_id}...")
        
        # Define essential files and directories for training
        essential_paths = [
            "src/train_autoencoder.py",
            "src/models/autoencoder.py",
            "src/dataset_maker/PairedPDFBuilder.py",
            "pyproject.toml",
            "requirements.txt"
        ]
        
        # Create a temporary directory to organize files for upload
        import tempfile
        import shutil
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Copy essential files to temp directory with directory structure
            for path in essential_paths:
                src_path = local_path / path
                if not src_path.exists():
                    print(f"Warning: {src_path} does not exist, skipping")
                    continue
                    
                # Create destination directory structure
                dest_path = temp_dir_path / path
                os.makedirs(dest_path.parent, exist_ok=True)
                
                # Copy the file
                shutil.copy2(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")
            
            # Upload the temp directory with essential files
            try:
                # Create a tar archive of the temp directory
                archive_path = Path(temp_dir) / "training_files.tar.gz"
                tar_cmd = ["tar", "-czf", str(archive_path), "-C", temp_dir, "."]
                subprocess.run(tar_cmd, check=True)
                print(f"Created archive at {archive_path}")
                
                # Step 1: Generate a one-time code by sending the archive
                print(f"Generating transfer code for archive...")
                send_cmd = ["runpodctl", "send", str(archive_path)]
                
                send_result = subprocess.run(
                    send_cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # Extract the one-time code from the output
                output = send_result.stdout
                print(f"Send output: {output}")
                
                # Try different regex patterns to extract the code
                import re
                code_match = None
                # Try patterns that RunPod CLI might use
                patterns = [
                    r'Code is: ([\w-]+)',  # Original pattern
                    r'Code:? ([\w-]+)',     # Alternative format
                    r'([A-Za-z0-9-]{6,})' # Just look for code-like string
                ]
                
                for pattern in patterns:
                    code_match = re.search(pattern, output)
                    if code_match:
                        break
                
                if not code_match:
                    print("Error: Could not extract transfer code from output")
                    return False
                    
                transfer_code = code_match.group(1)
                print(f"Generated transfer code: {transfer_code}")
                
                # Step 2: Run the receive command on the pod and extract the archive
                print(f"Receiving and extracting files on pod {self.pod_id}...")
                
                # Command to receive the file and extract it - make sure it's cleanly formatted
                # and each command is separate for better error handling
                pod_cmd = f"cd {self.volume_mount_path} && runpodctl receive {transfer_code} && ls -la && tar -xzf training_files.tar.gz && rm training_files.tar.gz"
                
                cd_cmd = [
                    "runpodctl", 
                    "start", 
                    "pod", 
                    self.pod_id, 
                    "--command", 
                    pod_cmd
                ]
                
                print(f"Running command on pod...")
                try:
                    cd_result = subprocess.run(
                        cd_cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=90  # Set a timeout to prevent hanging
                    )
                    
                    print(f"Pod command stdout: {cd_result.stdout}")
                    if cd_result.stderr:
                        print(f"Pod command stderr: {cd_result.stderr}")
                        
                    # Verify the files were properly extracted on the pod
                    print(f"Verifying files were extracted properly...")
                    verify_cmd = [
                        "runpodctl",
                        "start",
                        "pod",
                        self.pod_id,
                        "--command",
                        f"cd {self.volume_mount_path} && ls -la src/"
                    ]
                    
                    verify_result = subprocess.run(
                        verify_cmd,
                        check=False,  # Don't fail if this command fails
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if verify_result.returncode == 0:
                        print(f"Files on pod: {verify_result.stdout}")
                        if "train_autoencoder.py" in verify_result.stdout or "models" in verify_result.stdout:
                            print(f"Successfully uploaded essential files to pod {self.pod_id}:{self.volume_mount_path}")
                            return True
                        else:
                            print("Warning: Essential files not found in verification. Upload may have failed.")
                            return False
                    else:
                        print(f"Warning: Could not verify files: {verify_result.stderr}")
                        # Continue anyway since the main upload command succeeded
                        return True
                        
                except subprocess.TimeoutExpired:
                    print("Error: Command on pod timed out. This could indicate network issues or pod unresponsiveness.")
                    print("Please check the pod status manually and try again or follow manual instructions.")
                    return False
                except Exception as e:
                    print(f"Error executing command on pod: {e}")
                    return False
                    
            except subprocess.CalledProcessError as e:
                print(f"Error during file transfer: {e}")
                print(f"STDERR: {e.stderr}")
                return False
    
    def install_requirements(self) -> bool:
        """Install requirements on the pod.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.pod_id:
            raise ValueError("No pod ID available")
        
        print(f"Installing requirements on pod {self.pod_id}...")
        
        # Create a temporary script to run installation commands
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp_file:
            install_script = """
import subprocess
import sys

print("Installing basic ML requirements...")
try:
    subprocess.run(["pip", "install", "torch", "torchvision", "datasets", "matplotlib", "tqdm", "pydantic", "wandb"], 
                   check=True, capture_output=True, text=True)
    
    print("Checking for requirements.txt...")
    # Check if requirements.txt exists and install from it
    import os
    if os.path.exists("/workspace/requirements.txt"):
        subprocess.run(["pip", "install", "-r", "/workspace/requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("Installed dependencies from requirements.txt")
    
    # Install package in development mode if pyproject.toml exists
    if os.path.exists("/workspace/pyproject.toml"):
        subprocess.run(["pip", "install", "-e", "."], 
                      cwd="/workspace", check=True, capture_output=True, text=True)
        print("Installed package in development mode")
    
    print("All requirements installed successfully")
    sys.exit(0)
except Exception as e:
    print(f"Error installing requirements: {str(e)}")
    sys.exit(1)
"""
            tmp_file.write(install_script.encode())
            tmp_file.flush()
            tmp_path = tmp_file.name
        
        try:
            print("Running installation script on pod...")
            cmd = [
                "runpodctl", 
                "exec", 
                "python",
                tmp_path,
                "--pod_id",
                self.pod_id
            ]
            
            # Execute the installation script
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            if result.stdout:
                print(result.stdout)
            
            print("Successfully installed all requirements")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e}")
            if hasattr(e, 'stdout') and e.stdout:
                print(f"OUTPUT: {e.stdout}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"ERROR: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            print("Error: Command timed out while installing requirements")
            return False
        except Exception as e:
            print(f"Unexpected error installing requirements: {e}")
            return False
        finally:
            # Clean up the temporary script
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def start_training(
        self, 
        script_path: str = "src/train_autoencoder.py",
        args: Optional[List[str]] = None,
        use_wandb: bool = False,
        wandb_api_key: Optional[str] = None,
    ) -> str:
        """Start training on the pod.
        
        Args:
            script_path: Path to training script (relative to project root)
            args: Additional arguments to pass to the script
            use_wandb: Whether to use Weights & Biases
            wandb_api_key: Weights & Biases API key
            
        Returns:
            Command ID
        """
        if not self.pod_id:
            raise ValueError("No pod ID available")
        
        print(f"Starting training on pod {self.pod_id}...")
        
        # Prepare arguments and environment variables
        args = args or []
        train_args = []
        
        if use_wandb:
            train_args.append("--wandb")
            wandb_api_key = wandb_api_key or os.environ.get("WANDB_API_KEY")
            if wandb_api_key:
                train_args.append(f"--wandb-key {wandb_api_key}")
        
        # Add any additional arguments
        train_args.extend(args)
        
        # Create a temporary script that will run the training
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp_file:
            train_launcher_script = f"""
import subprocess
import sys
import os
import time

print("Starting training script execution...")

try:
    # Change to workspace directory
    os.chdir("/workspace")
    
    # Build command
    cmd = ["python", "{script_path}"] + {train_args}
    
    print(f"Running command: {{' '.join(cmd)}}")
    
    # Run the training script and stream output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end="")
        
    # Wait for process to complete
    returncode = process.wait()
    if returncode != 0:
        print(f"Training exited with non-zero return code: {{returncode}}")
        sys.exit(returncode)
    
    print("Training completed successfully!")
    sys.exit(0)
    
except Exception as e:
    print(f"Error executing training: {{str(e)}}")
    sys.exit(1)
"""
            tmp_file.write(train_launcher_script.encode())
            tmp_file.flush()
            tmp_path = tmp_file.name
    
        try:
            # Execute the training script on the pod
            print(f"Executing training script {script_path} on pod {self.pod_id}...")
            cmd = [
                "runpodctl", 
                "exec", 
                "python",
                tmp_path,
                "--pod_id",
                self.pod_id
            ]
            
            # Start the process and stream output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            # Print output in real-time
            print("\nTraining started. Output:")
            print("-" * 50)
            
            # Set up timeout for SSH connection
            start_time = time.time()
            timeout = 60  # 60 seconds timeout for SSH connection
            waiting_for_connection = True
            connection_message_shown = False
            
            try:
                for line in process.stdout:
                    # Check for SSH connection message
                    if "Waiting for Pod to come online" in line and waiting_for_connection:
                        if not connection_message_shown:
                            print("Attempting to establish SSH connection to the pod...")
                            print("This may take a minute, especially on newly started pods.")
                            connection_message_shown = True
                        
                        # Check for timeout
                        if time.time() - start_time > timeout:
                            print("SSH connection timed out. The pod might be unreachable or still initializing.")
                            print("Consider checking pod status or trying again in a few minutes.")
                            raise TimeoutError("SSH connection timed out")
                    elif line.strip() and waiting_for_connection:
                        # We got some output, so connection is established
                        waiting_for_connection = False
                        
                    print(line, end="")
                
                return self.pod_id
                
            except KeyboardInterrupt:
                print("\nInterrupted by user. Training continues on RunPod.")
                print("You can check the status later with the RunPod console.")
                return self.pod_id
                
            except Exception as e:
                print(f"Error starting training: {e}")
                
                # Provide manual instructions as fallback
                cmd_str = f"cd /workspace && python {script_path} {' '.join(train_args)}"
                
                print(f"\n{'=' * 80}")
                print(f"MANUAL TRAINING INSTRUCTIONS FOR POD {self.pod_id}")
                print(f"{'=' * 80}")
                print("\nAn error occurred while trying to start training remotely.")
                print("Please connect to your pod via SSH or Web Terminal and run:")
                print("\n    " + cmd_str)
                print(f"{'=' * 80}\n")
                
                return self.pod_id
        finally:
            # Clean up the temporary script
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def stop_pod(self) -> bool:
        """Stop the pod.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.pod_id:
            raise ValueError("No pod ID available")
        
        print(f"Stopping pod {self.pod_id}...")
        
        cmd = ["runpodctl", "stop", "pod", self.pod_id]
        self._run_command(cmd)
        
        print(f"Pod {self.pod_id} stopped")
        return True
    
    def terminate_pod(self) -> bool:
        """Terminate the pod.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.pod_id:
            raise ValueError("No pod ID available")
        
        print(f"Terminating pod {self.pod_id}...")
        
        cmd = ["runpodctl", "terminate", "pod", self.pod_id]
        self._run_command(cmd)
        
        print(f"Pod {self.pod_id} terminated")
        return True


def print_manual_instructions():
    """Print manual instructions for RunPod setup when automation fails."""
    print("\n" + "=" * 80)
    print("MANUAL RUNPOD SETUP INSTRUCTIONS")
    print("=" * 80)
    print(
        "\nIf the automated RunPod management is failing, follow these manual steps:\n"
        "\n1. CREATE POD:\n"
        "   - Go to https://runpod.io/console/pods\n"
        "   - Click 'Deploy' and select a GPU\n"
        "   - Choose 'PyTorch' template\n"
        "   - Set container disk to at least 10GB\n"
        "   - Set volume size to at least 20GB\n"
        "   - Click 'Deploy'\n"
        "\n2. UPLOAD CODE:\n"
        "   - Connect to your pod via SSH or use the Web Terminal\n"
        "   - Create a directory: mkdir -p /workspace/document-cleaning\n"
        "   - Use the RunPod file browser to upload these essential files:\n"
        "     * src/train_autoencoder.py\n"
        "     * src/models/autoencoder.py\n"
        "     * src/dataset_maker/PairedPDFBuilder.py\n"
        "     * pyproject.toml\n"
        "\n3. INSTALL DEPENDENCIES:\n"
        "   - Run: pip install -e .\n"
        "   - Run: pip install wandb\n"
        "\n4. RUN TRAINING:\n"
        "   - To train with wandb: python src/train_autoencoder.py --wandb\n"
        "   - To train without wandb: python src/train_autoencoder.py\n"
        "\n5. WHEN FINISHED:\n"
        "   - Stop your pod from the RunPod console to avoid charges\n"
    )
    print("=" * 80 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run training on RunPod")
    
    # RunPod configuration
    parser.add_argument("--api-key", help="RunPod API key")
    parser.add_argument("--gpu-type", default="NVIDIA RTX A4000", help="GPU type to use")
    parser.add_argument("--template", default="runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel", help="Docker template ID")
    parser.add_argument("--container-disk", type=int, default=10, help="Container disk size in GB")
    parser.add_argument("--volume-disk", type=int, default=20, help="Volume disk size in GB")
    parser.add_argument("--name", help="Name for the pod")
    
    # Actions
    parser.add_argument("--create", action="store_true", help="Create a new pod")
    parser.add_argument("--upload", help="Upload local directory to pod")
    parser.add_argument("--install", action="store_true", help="Install requirements")
    parser.add_argument("--train", action="store_true", help="Start training")
    parser.add_argument("--stop", action="store_true", help="Stop the pod")
    parser.add_argument("--terminate", action="store_true", help="Terminate the pod")
    parser.add_argument("--pod-id", help="Existing pod ID to use")
    parser.add_argument("--manual", action="store_true", help="Show manual instructions")
    
    # Training configuration
    parser.add_argument("--script", default="src/train_autoencoder.py", help="Training script path")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb-key", help="Weights & Biases API key")
    
    # Pass remaining args to the training script
    parser.add_argument("training_args", nargs="*", help="Arguments to pass to the training script")
    
    args = parser.parse_args()
    
    # Show manual instructions if requested
    if args.manual:
        print_manual_instructions()
        return
    
    # Check if we can connect to RunPod API
    try:
        result = subprocess.run(["runpodctl", "get", "pod"], 
                      check=False, capture_output=True, text=True)
        if "Error: data is nil" in result.stderr:
            print("RunPod CLI authentication is failing. Please check your API key setup.")
            print("Run 'runpodctl config set apiKey YOUR_API_KEY' to set your API key.")
            print("\nFalling back to manual instructions:")
            print_manual_instructions()
            return
    except Exception as e:
        print(f"Error connecting to RunPod API: {e}")
        print("\nRunPod CLI authentication may be failing. Please check your API key setup.")
        print("Run 'runpodctl config set apiKey YOUR_API_KEY' to set your API key.")
        print("\nFalling back to manual instructions:")
        print_manual_instructions()
        return
    
    # Initialize trainer
    trainer = RunpodTrainer(
        api_key=args.api_key,
        gpu_type_id=args.gpu_type,
        template_id=args.template,
        container_disk_size_gb=args.container_disk,
        volume_disk_size_gb=args.volume_disk,
        name=args.name,
    )
    
    # Use existing pod if provided
    if args.pod_id:
        trainer.pod_id = args.pod_id
    
    # Execute actions
    if args.create:
        try:
            trainer.create_pod()
            trainer.wait_for_pod_ready()
        except Exception as e:
            print(f"Error creating pod: {e}")
            print("\nFalling back to manual instructions:")
            print_manual_instructions()
    
    if args.upload:
        try:
            trainer.upload_code(args.upload)
        except Exception as e:
            print(f"Error uploading code: {e}")
            print("\nFalling back to manual instructions for file upload:")
            print_manual_instructions()
    
    if args.install:
        try:
            trainer.install_requirements()
        except Exception as e:
            print(f"Error installing requirements: {e}")
            print("\nPlease install requirements manually:")
            print("  1. Connect to your pod via SSH or Web Terminal")
            print("  2. Run: pip install -e .")
            print("  3. Run: pip install wandb")
    
    if args.train:
        try:
            trainer.start_training(
                script_path=args.script,
                args=args.training_args,
                use_wandb=args.wandb,
                wandb_api_key=args.wandb_key,
            )
        except Exception as e:
            print(f"Error running training: {e}")
            print("\nPlease run training manually:")
            print("  1. Connect to your pod via SSH or Web Terminal")
            print("  2. Run: cd /workspace")
            print("  3. Run: python src/train_autoencoder.py" + (" --wandb" if args.wandb else ""))
    
    if args.stop:
        try:
            trainer.stop_pod()
        except Exception as e:
            print(f"Error stopping pod: {e}")
            print("\nPlease stop your pod manually from the RunPod console.")
    
    if args.terminate:
        try:
            trainer.terminate_pod()
        except Exception as e:
            print(f"Error terminating pod: {e}")
            print("\nPlease terminate your pod manually from the RunPod console.")


if __name__ == "__main__":
    main()

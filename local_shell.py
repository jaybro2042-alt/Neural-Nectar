from dataclasses import dataclass
import subprocess, os, sys, shlex
from typing import Dict, Any

# If you have your own decorator, import it here:
try:
    from agents.tool import function_tool  # your repo's decorator path
except Exception:
    # Fallback no-op decorator so this file runs standalone in demos
    def function_tool(*args, **kwargs):
        def wrap(f): return f
        return wrap

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

@dataclass
class LocalShellExecutor:
    shell: str = "powershell"  # "powershell" | "pwsh" | "bash"
    timeout: int = 180

    def run(self, cmd: str) -> Dict[str, Any]:
        if not cmd or not cmd.strip():
            return {"ok": False, "error": "empty command"}

        if self.shell.lower().startswith("power"):
            full = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd]
        else:
            full = ["bash", "-lc", cmd]

        try:
            p = subprocess.run(
                full, cwd=PROJECT_ROOT,
                capture_output=True, text=True, timeout=self.timeout
            )
            return {
                "ok": p.returncode == 0,
                "returncode": p.returncode,
                "stdout": p.stdout[-10000:],  # tail to keep payload small
                "stderr": p.stderr[-10000:]
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {self.timeout}s"}

executor = LocalShellExecutor()

@function_tool(name_override="local_shell",
               description_override="Execute a shell command in the project root and return stdout/stderr/exit code.",
               strict_mode=True,
               is_enabled=True)  # your master_config/tool-gates can further control this
def local_shell(command: str) -> Dict[str, Any]:
    """
    Run a shell command from the repository root.

    Args:
        command: The shell command to execute.
    Returns:
        { ok, returncode, stdout, stderr }
    """
    return executor.run(command)

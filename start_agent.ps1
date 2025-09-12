Set-Location "C:\Users\Casey\Downloads\Trading algo\api_agent_project"
.\.venv\Scripts\Activate.ps1
$env:OPENAI_API_KEY = [Environment]::GetEnvironmentVariable("OPENAI_API_KEY","User")
$env:AGENTS_LOCAL_SHELL = "wsl"
py -3 .\main_agent_entry.py
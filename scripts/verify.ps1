$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    uv sync
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed with exit code $LASTEXITCODE" }

    uv run ruff check .
    if ($LASTEXITCODE -ne 0) { throw "ruff check failed with exit code $LASTEXITCODE" }

    uv run ruff format --check .
    if ($LASTEXITCODE -ne 0) { throw "ruff format check failed with exit code $LASTEXITCODE" }

    uv run pytest
    if ($LASTEXITCODE -ne 0) { throw "pytest failed with exit code $LASTEXITCODE" }

    uv run ddareungi doctor
    if ($LASTEXITCODE -ne 0) { throw "environment doctor failed with exit code $LASTEXITCODE" }

    git diff --check
    if ($LASTEXITCODE -ne 0) { throw "git diff check failed with exit code $LASTEXITCODE" }

    $candidateFiles = git ls-files --cached --others --exclude-standard
    if ($LASTEXITCODE -ne 0) { throw "git file listing failed with exit code $LASTEXITCODE" }

    $textCandidateFiles = @()
    foreach ($relativePath in $candidateFiles) {
        if (-not (Test-Path -LiteralPath $relativePath -PathType Leaf)) { continue }

        $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $relativePath))
        if ($bytes -notcontains 0) {
            $textCandidateFiles += $relativePath
        }
    }

    $whitespaceErrors = @()
    foreach ($relativePath in $textCandidateFiles) {
        $lineNumber = 0
        Get-Content -LiteralPath $relativePath -Encoding utf8 | ForEach-Object {
            $lineNumber += 1
            if ($_ -match '[ \t]+$') {
                $whitespaceErrors += "${relativePath}:${lineNumber}"
            }
        }
    }

    if ($whitespaceErrors.Count -gt 0) {
        throw "Trailing whitespace found:`n$($whitespaceErrors -join "`n")"
    }

    if (Test-Path -LiteralPath '.env' -PathType Leaf) {
        $apiKeyLine = Get-Content -LiteralPath '.env' -Encoding utf8 |
            Where-Object { $_ -match '^SEOUL_OPEN_DATA_API_KEY=' } |
            Select-Object -First 1
        $apiKey = ''
        if ($apiKeyLine) {
            $apiKey = ($apiKeyLine -split '=', 2)[1].Trim()
        }

        if ($apiKey) {
            $exposedFiles = @()
            foreach ($relativePath in $textCandidateFiles) {
                $contents = Get-Content -LiteralPath $relativePath -Raw -Encoding utf8
                if ($contents.Contains($apiKey)) {
                    $exposedFiles += $relativePath
                }
            }

            if ($exposedFiles.Count -gt 0) {
                throw "API key found in Git candidate files: $($exposedFiles -join ', ')"
            }
        }
    }
}
finally {
    Pop-Location
}

param(
    [string]$RepoName = "agentic-safety-tutorial",
    [string]$Visibility = "public",
    [string]$UserName = "",
    [string]$UserEmail = "",
    [switch]$CommitIfNeeded
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$gh = "C:\Program Files\GitHub CLI\gh.exe"
$git = "C:\Program Files\Git\bin\git.exe"

if (-not (Test-Path $gh)) {
    throw "GitHub CLI not found at $gh"
}
if (-not (Test-Path $git)) {
    throw "git not found at $git"
}

& $gh auth status | Out-Null

$hasHead = $true
try {
    & $git -C $repoRoot rev-parse --verify HEAD | Out-Null
}
catch {
    $hasHead = $false
}

$statusOutput = & $git -C $repoRoot status --porcelain
if ($statusOutput) {
    if (-not $CommitIfNeeded) {
        Write-Host "Working tree is not clean. Re-run with -CommitIfNeeded or commit manually before publishing." -ForegroundColor Yellow
        Write-Host $statusOutput
        exit 1
    }
    if (-not $UserName -or -not $UserEmail) {
        throw "CommitIfNeeded requires -UserName and -UserEmail so the initial commit can be created safely."
    }
    & $git -C $repoRoot config user.name $UserName
    & $git -C $repoRoot config user.email $UserEmail
    & $git -C $repoRoot add .
    $commitMessage = if ($hasHead) { "Prepare tutorial repo for GitHub publish" } else { "Initial tutorial repo import" }
    & $git -C $repoRoot commit -m $commitMessage
}

$createArgs = @(
    "repo",
    "create",
    $RepoName,
    "--source",
    $repoRoot,
    "--remote",
    "origin",
    "--push",
    "--$Visibility"
)

& $gh @createArgs

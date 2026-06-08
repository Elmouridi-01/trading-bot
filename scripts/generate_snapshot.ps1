$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Set-Location $root

$excludeDirs = @('venv', '__pycache__', '.git', '.pytest_cache', 'node_modules')
$excludeExt = @('.pkl', '.db', '.env', '.db-shm', '.db-wal')
$excludeFiles = @('PROJECT_FULL_SNAPSHOT.md')
$excludePathParts = @('\logs\', '\.pytest_cache\')

$files = Get-ChildItem -Recurse -File | Where-Object {
    $p = $_.FullName
    if ($excludeFiles -contains $_.Name) { return $false }
    if ($excludeExt -contains $_.Extension) { return $false }
    if ($_.Name -match '\.db(-shm|-wal)?$') { return $false }
    foreach ($d in $excludeDirs) {
        if ($p -match "[\\/]$([regex]::Escape($d))[\\/]" -or $p -match "[\\/]$([regex]::Escape($d))$") { return $false }
    }
    foreach ($part in $excludePathParts) {
        if ($p -like "*$part*") { return $false }
    }
    if ($_.Extension -in @('.log', '.log.1')) { return $false }
    if ($_.Name -eq '.env') { return $false }
    return $true
} | Sort-Object FullName

function Get-Lang([string]$path) {
    $name = [IO.Path]::GetFileName($path)
    if ($name -eq 'Dockerfile') { return 'dockerfile' }
    switch ([IO.Path]::GetExtension($path).ToLower()) {
        '.py' { 'python' }
        '.yml' { 'yaml' }
        '.yaml' { 'yaml' }
        '.toml' { 'toml' }
        '.ini' { 'ini' }
        '.md' { 'markdown' }
        '.txt' { 'text' }
        '.csv' { 'csv' }
        default { 'text' }
    }
}

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('# PROJECT FULL SNAPSHOT')
$lines.Add('')
$lines.Add('> Auto-generated snapshot of the entire codebase.')
$lines.Add("> Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add('> Excluded: venv, __pycache__, .env, *.pkl, *.db, logs, .pytest_cache')
$lines.Add('')
$lines.Add('## Table of Contents')
$lines.Add('')

foreach ($f in $files) {
    $rel = $f.FullName.Substring($root.Length + 1).Replace('\', '/')
    $anchor = ($rel.ToLower() -replace '[^a-z0-9]+', '-').Trim('-')
    $lines.Add("- [$rel](#$anchor)")
}

$lines.Add('')
$lines.Add('---')
$lines.Add('')

foreach ($f in $files) {
    $rel = $f.FullName.Substring($root.Length + 1).Replace('\', '/')
    $lang = Get-Lang $f.FullName
    $lines.Add("## File: ``$rel``")
    $lines.Add('')
    $lines.Add("``````$lang")
    $content = Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
    if ($null -eq $content) { $content = '' }
    foreach ($line in ($content.TrimEnd() -split "`n")) {
        $lines.Add($line.TrimEnd("`r"))
    }
    $lines.Add('``````')
    $lines.Add('')
    $lines.Add('---')
    $lines.Add('')
}

$outPath = Join-Path $root 'PROJECT_FULL_SNAPSHOT.md'
[System.IO.File]::WriteAllLines($outPath, $lines, [System.Text.UTF8Encoding]::new($false))
Write-Host "Written: $outPath"
Write-Host "Files included: $($files.Count)"
Write-Host "Size: $((Get-Item $outPath).Length) bytes"

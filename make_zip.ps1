# Builds the zip to hand to teammates:  .\make_zip.ps1
#
# Writes dist\UOS_Chatbot.zip with a short top folder ("UOS_Chatbot") so that unzipping never runs into the Windows
# 260-character path limit. Left out on purpose:
#   .env, var\            your Groq key, the saved conversations and the cookie secret
#   .git, archive\        history and the earlier prototypes
#   android build output  generated files with the longest paths in the repository, and local.properties
#   __pycache__, stale cache files, dist\
# The current embedding cache IS included, so a teammate's first start does not have to rebuild the index.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$top = "UOS_Chatbot"
$stage = Join-Path $env:TEMP ("uos_zip_" + [guid]::NewGuid().ToString("N").Substring(0, 8))
$dest = Join-Path $stage $top
New-Item -ItemType Directory -Force $dest | Out-Null

$excludeDirs = @(".git", "archive", "var", "dist", "__pycache__", ".pytest_cache", ".venv", ".venv_broken_backup",
                 ".nicegui", "backups", ".gradle", ".kotlin", ".idea", "build", "captures")
$excludeFiles = @(".env", "local.properties", "*.pyc", "*.apk", "*.jks", "*.keystore", "*.db", "*.db-wal", "*.db-shm")
robocopy $PSScriptRoot $dest /E /XD $excludeDirs /XF $excludeFiles /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }

# Keep only the newest embedding cache set (the files the last start used).
$cache = Join-Path $dest "cache"
if (Test-Path $cache) {
    # "the last one read" is decided in the project's own cache folder: the copies in the staging folder are all new
    $newest = Get-ChildItem (Join-Path $PSScriptRoot "cache") -Filter "*_embeddings.npy" |
        Sort-Object LastAccessTime, LastWriteTime -Descending | Select-Object -First 1
    if ($newest) {
        $prefix = $newest.Name -replace "_embeddings\.npy$", ""
        Get-ChildItem $cache -File | Where-Object { -not $_.Name.StartsWith($prefix + "_") } | Remove-Item -Force
    }
}

$longest = (Get-ChildItem $dest -Recurse -File | ForEach-Object { $_.FullName.Length - $stage.Length - 1 } | Measure-Object -Maximum).Maximum
New-Item -ItemType Directory -Force (Join-Path $PSScriptRoot "dist") | Out-Null
$zip = Join-Path $PSScriptRoot "dist\$top.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
# Entries are written one by one with "/" separators: Windows PowerShell's CreateFromDirectory writes "\", which
# macOS and Linux unzip as file names containing backslashes instead of folders.
$out = [IO.Compression.ZipFile]::Open($zip, [IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($file in Get-ChildItem $stage -Recurse -File) {
        $name = $file.FullName.Substring($stage.Length + 1).Replace("\", "/")
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($out, $file.FullName, $name, [IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
} finally {
    $out.Dispose()
}
Remove-Item $stage -Recurse -Force

$archive = [IO.Compression.ZipFile]::OpenRead($zip)
$files = $archive.Entries.Count
$archive.Dispose()
"{0}  {1:N1} MB  {2} files  longest path inside the zip: {3} characters" -f $zip, ((Get-Item $zip).Length / 1MB), $files, $longest

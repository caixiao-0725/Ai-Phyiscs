param(
    [string]$Executable = "",
    [string]$OutputCsv = "",
    [int]$Repeats = 3,
    [int]$DebugInterval = 1,
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) {
    $Executable = Join-Path $repoRoot "build/Release/chysx_scene.exe"
}
if ([string]::IsNullOrWhiteSpace($OutputCsv)) {
    $OutputCsv = Join-Path $repoRoot "output/pabd_contact_stiffness_experiment.csv"
}
if (-not (Test-Path -LiteralPath $Executable)) {
    throw "Scene executable not found: $Executable"
}

function Number-Token([string]$Line, [string]$Name) {
    $pattern = "(?:^|\s)" + [regex]::Escape($Name) + "=([^\s]+)"
    $match = [regex]::Match($Line, $pattern)
    if (-not $match.Success) { return [double]::NaN }
    return [double]::Parse(
        $match.Groups[1].Value,
        [System.Globalization.NumberStyles]::Float,
        $invariant)
}

function Mean-Property($Rows, [string]$Name) {
    if ($Rows.Count -eq 0) { return [double]::NaN }
    return [double](($Rows | Measure-Object -Property $Name -Average).Average)
}

function Max-Property($Rows, [string]$Name) {
    if ($Rows.Count -eq 0) { return [double]::NaN }
    return [double](($Rows | Measure-Object -Property $Name -Maximum).Maximum)
}

function Parse-Gear-Final($Lines) {
    $line = @($Lines | Where-Object { $_ -match '^\[PABD_GEARS frame=' }) |
        Select-Object -Last 1
    if ($null -eq $line) {
        return [pscustomobject]@{
            AxisW0 = [double]::NaN
            AxisW1 = [double]::NaN
            SpeedRatio = [double]::NaN
            MaxEndpointErr = [double]::NaN
        }
    }
    $axis0 = [double]::NaN
    $axis1 = [double]::NaN
    $speedRatio = [double]::NaN
    $maxEndpoint = [double]::NaN
    $axisMatch = [regex]::Match($line, 'axisW=\[([^\]]*)\]')
    if ($axisMatch.Success) {
        $values = @($axisMatch.Groups[1].Value.Split(',') |
            ForEach-Object { [double]::Parse($_, $invariant) })
        if ($values.Count -gt 0) { $axis0 = $values[0] }
        if ($values.Count -gt 1) { $axis1 = $values[1] }
        if ($values.Count -gt 1 -and [math]::Abs($axis0) -gt 1.0e-20) {
            $speedRatio = [math]::Abs($axis1 / $axis0)
        }
    }
    $endpointMatch = [regex]::Match($line, 'endpointError=\[([^\]]*)\]')
    if ($endpointMatch.Success) {
        $values = @($endpointMatch.Groups[1].Value.Split(',') |
            ForEach-Object { [math]::Abs([double]::Parse($_, $invariant)) })
        if ($values.Count -gt 0) {
            $maxEndpoint = [double](($values | Measure-Object -Maximum).Maximum)
        }
    }
    return [pscustomobject]@{
        AxisW0 = $axis0
        AxisW1 = $axis1
        SpeedRatio = $speedRatio
        MaxEndpointErr = $maxEndpoint
    }
}

$cases = @()
foreach ($ratio in @(1.0, 2.0, 5.0, 10.0)) {
    $cases += [pscustomobject]@{
        Name = "stack-volume-r$ratio"
        Scene = "CUDA PABD: Stacked Blocks"
        Measure = "tetra_volume"
        PfScale = $ratio
        SelfBeta = 16.0
        Frames = 600
        Warmup = 180
    }
}
$cases += [pscustomobject]@{
    Name = "stack-effective-raw-beta"
    Scene = "CUDA PABD: Stacked Blocks"
    Measure = "effective_mass"
    PfScale = 1.0
    SelfBeta = 16.0
    Frames = 600
    Warmup = 180
}
$cases += [pscustomobject]@{
    Name = "stack-effective-calibrated"
    Scene = "CUDA PABD: Stacked Blocks"
    Measure = "effective_mass"
    PfScale = 1.0
    SelfBeta = 1.25
    Frames = 600
    Warmup = 180
}
foreach ($ratio in @(1.0, 5.0)) {
    $cases += [pscustomobject]@{
        Name = "ee-cross-volume-r$ratio"
        Scene = "CUDA PABD: Tetra EE Cross"
        Measure = "tetra_volume"
        PfScale = $ratio
        SelfBeta = 16.0
        Frames = 360
        Warmup = 120
    }
}

if (-not $Quick) {
    foreach ($ratio in @(1.0, 2.0, 5.0, 10.0)) {
        $cases += [pscustomobject]@{
            Name = "chain4-volume-r$ratio"
            Scene = "CUDA PABD: Rigid-IPC Chain Net 4x4"
            Measure = "tetra_volume"
            PfScale = $ratio
            SelfBeta = 16.0
            Frames = 600
            Warmup = 180
        }
        $cases += [pscustomobject]@{
            Name = "gear2-volume-r$ratio"
            Scene = "CUDA PABD: Torque Gear Line 2"
            Measure = "tetra_volume"
            PfScale = $ratio
            SelfBeta = 16.0
            Frames = 600
            Warmup = 180
        }
    }
}

$oldMeasure = $env:CHYSX_PABD_SELF_CONTACT_MEASURE
$oldPfScale = $env:CHYSX_PABD_PF_STIFFNESS_SCALE
$oldSelfBeta = $env:CHYSX_PABD_SELF_BETA
$oldDebugInterval = $env:CHYSX_PABD_DEBUG_INTERVAL
$oldFeatures = $env:CHYSX_PABD_CONTACT_FEATURES
$results = @()

try {
    foreach ($case in $cases) {
        foreach ($repeat in 1..([math]::Max(1, $Repeats))) {
            $env:CHYSX_PABD_SELF_CONTACT_MEASURE = $case.Measure
            $env:CHYSX_PABD_PF_STIFFNESS_SCALE =
                $case.PfScale.ToString("0.###", $invariant)
            $env:CHYSX_PABD_SELF_BETA =
                $case.SelfBeta.ToString("0.###", $invariant)
            $env:CHYSX_PABD_DEBUG_INTERVAL =
                ([math]::Max(1, $DebugInterval)).ToString($invariant)
            $features = if ($case.Scene -eq "CUDA PABD: Tetra EE Cross") {
                "ee"
            } else { "both" }
            $env:CHYSX_PABD_CONTACT_FEATURES = $features

            Write-Host ("[contact experiment] {0} repeat={1}" -f
                $case.Name, $repeat)
            $lines = @(& $Executable --scene $case.Scene --frames $case.Frames `
                --timing-warmup $case.Warmup --no-export 2>&1 |
                ForEach-Object { "$_" })
            $exitCode = $LASTEXITCODE

            $allSamples = @()
            foreach ($line in $lines) {
                $frameMatch = [regex]::Match(
                    $line, '^\[PABD_DBG frame=(\d+)\].*maxPen=')
                if (-not $frameMatch.Success) { continue }
                $frame = [int]$frameMatch.Groups[1].Value
                $allSamples += [pscustomobject]@{
                    Frame = $frame
                    ActivePf = Number-Token $line "activePf"
                    ActiveEe = Number-Token $line "activeEe"
                    MaxPen = Number-Token $line "maxPen"
                    RmsPen = Number-Token $line "rmsPen"
                    MeanK = Number-Token $line "meanK"
                    PfK = Number-Token $line "pfK"
                    EeK = Number-Token $line "eeK"
                    PfV = Number-Token $line "pfV"
                    EeV = Number-Token $line "eeV"
                    MinY = Number-Token $line "minY"
                }
            }
            $samples = @($allSamples | Where-Object {
                $_.Frame -ge $case.Warmup
            })

            $setupLine = @($lines | Where-Object {
                $_ -match '^\[PABD_DBG setup\]'
            }) | Select-Object -Last 1
            $timingLine = @($lines | Where-Object {
                $_ -match '^\[TIMING summary\]'
            }) | Select-Object -Last 1
            $metricsLine = @($lines | Where-Object {
                $_ -match '^\[METRICS summary\]'
            }) | Select-Object -Last 1
            $gear = Parse-Gear-Final $lines
            $contactGap = if ($null -ne $setupLine) {
                Number-Token $setupLine "gap"
            } else { [double]::NaN }
            $maxPenAll = Max-Property $allSamples "MaxPen"
            $maxPenSteady = Max-Property $samples "MaxPen"
            $meanRmsPen = Mean-Property $samples "RmsPen"
            $hasOverflow = @($lines | Where-Object {
                $_ -match '^\[PABD_ERROR\]' -or
                $_ -match '(?:^|\s)overflow=1(?:\s|$)' -or
                $_ -match '(?:^|\s)boverflow=1(?:\s|$)'
            }).Count -gt 0

            $results += [pscustomobject]@{
                Name = $case.Name
                Repeat = $repeat
                Scene = $case.Scene
                Features = $features
                Measure = $case.Measure
                PfScale = $case.PfScale
                SelfBeta = $case.SelfBeta
                Frames = $case.Frames
                AllSamples = $allSamples.Count
                Samples = $samples.Count
                ContactGap = $contactGap
                MaxPen = $maxPenAll
                MaxPenSteady = $maxPenSteady
                MeanRmsPen = $meanRmsPen
                MaxPenOverGap = if ($contactGap -gt 0.0) {
                    $maxPenAll / $contactGap
                } else { [double]::NaN }
                MeanRmsPenOverGap = if ($contactGap -gt 0.0) {
                    $meanRmsPen / $contactGap
                } else { [double]::NaN }
                MaxActivePfAll = Max-Property $allSamples "ActivePf"
                MaxActiveEeAll = Max-Property $allSamples "ActiveEe"
                MeanK = Mean-Property $samples "MeanK"
                MeanPfK = Mean-Property $samples "PfK"
                MeanEeK = Mean-Property $samples "EeK"
                MeanPfVolume = Mean-Property $samples "PfV"
                MeanEeVolume = Mean-Property $samples "EeV"
                MeanActivePf = Mean-Property $samples "ActivePf"
                MeanActiveEe = Mean-Property $samples "ActiveEe"
                FinalMinY = if ($allSamples.Count -gt 0) {
                    $allSamples[-1].MinY
                } else { [double]::NaN }
                FinalAxisW0 = $gear.AxisW0
                FinalAxisW1 = $gear.AxisW1
                FinalGearSpeedRatio = $gear.SpeedRatio
                MaxEndpointErr = $gear.MaxEndpointErr
                MeanMs = if ($null -ne $timingLine) {
                    Number-Token $timingLine "meanMs"
                } else { [double]::NaN }
                P95Ms = if ($null -ne $timingLine) {
                    Number-Token $timingLine "p95Ms"
                } else { [double]::NaN }
                Nan = if ($null -ne $metricsLine) {
                    [int](Number-Token $metricsLine "nan")
                } else { -1 }
                Overflow = [int]$hasOverflow
                ExitCode = $exitCode
            }
        }
    }
} finally {
    if ($null -eq $oldMeasure) {
        Remove-Item Env:CHYSX_PABD_SELF_CONTACT_MEASURE -ErrorAction SilentlyContinue
    } else { $env:CHYSX_PABD_SELF_CONTACT_MEASURE = $oldMeasure }
    if ($null -eq $oldPfScale) {
        Remove-Item Env:CHYSX_PABD_PF_STIFFNESS_SCALE -ErrorAction SilentlyContinue
    } else { $env:CHYSX_PABD_PF_STIFFNESS_SCALE = $oldPfScale }
    if ($null -eq $oldSelfBeta) {
        Remove-Item Env:CHYSX_PABD_SELF_BETA -ErrorAction SilentlyContinue
    } else { $env:CHYSX_PABD_SELF_BETA = $oldSelfBeta }
    if ($null -eq $oldDebugInterval) {
        Remove-Item Env:CHYSX_PABD_DEBUG_INTERVAL -ErrorAction SilentlyContinue
    } else { $env:CHYSX_PABD_DEBUG_INTERVAL = $oldDebugInterval }
    if ($null -eq $oldFeatures) {
        Remove-Item Env:CHYSX_PABD_CONTACT_FEATURES -ErrorAction SilentlyContinue
    } else { $env:CHYSX_PABD_CONTACT_FEATURES = $oldFeatures }
}

$outputDirectory = Split-Path -Parent $OutputCsv
if (-not [string]::IsNullOrWhiteSpace($outputDirectory)) {
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
}
$results | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $OutputCsv
$results | Format-Table Name, Repeat, Measure, PfScale, SelfBeta, MaxPen, `
    MeanRmsPen, MeanActivePf, MeanActiveEe, MeanMs, Nan, Overflow -AutoSize
Write-Host "CSV: $OutputCsv"

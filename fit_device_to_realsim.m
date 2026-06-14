function result = fit_device_to_realsim(ltpFile, ltdFile, varargin)
%FIT_DEVICE_TO_REALSIM Fit LTP/LTD device curves and update RealDevice.
%
% Usage:
%   result = fit_device_to_realsim("device_LTP.csv", "device_LTD.csv")
%   result = fit_device_to_realsim("ltp.xlsx", "ltd.xlsx", ...
%       "ReadVoltage", 0.5, "WriteVoltageLTP", 3.2, "WriteVoltageLTD", 2.8, ...
%       "WritePulseWidthLTP", 300e-6, "WritePulseWidthLTD", 300e-6)
%
% Input files must contain at least two numeric columns:
%   column 1: pulse number
%   column 2: conductance in Siemens
%
% LTP should generally increase; LTD should generally decrease.

opts = struct();
opts.CellCpp = fullfile(fileparts(mfilename("fullpath")), "Cell.cpp");
opts.ReadVoltage = 0.5;
opts.WriteVoltageLTP = 3.2;
opts.WriteVoltageLTD = 2.8;
opts.WritePulseWidthLTP = 300e-6;
opts.WritePulseWidthLTD = 300e-6;
opts.ApplyToCellCpp = true;
opts.ShowFigure = true;
opts.SaveFigure = true;
opts.ReportFile = fullfile(fileparts(mfilename("fullpath")), "fit_device_result.txt");
opts = parseNameValue(opts, varargin{:});

if nargin < 1 || strlength(string(ltpFile)) == 0
    ltpFile = fullfile(fileparts(mfilename("fullpath")), "device_LTP.csv");
end
if nargin < 2 || strlength(string(ltdFile)) == 0
    ltdFile = fullfile(fileparts(mfilename("fullpath")), "device_LTD.csv");
end

ltp = readCurve(ltpFile);
ltd = readCurve(ltdFile);

gMin = min([ltp(:,2); ltd(:,2)]);
gMax = max([ltp(:,2); ltd(:,2)]);
if ~(isfinite(gMin) && isfinite(gMax) && gMax > gMin && gMin >= 0)
    error("Invalid conductance range. Check that conductance values are positive and not constant.");
end

ltpX = normalizePulse(ltp(:,1));
ltpY = clamp01((ltp(:,2) - gMin) / (gMax - gMin));
ltdX = normalizePulse(ltd(:,1));
ltdY = clamp01((ltd(:,2) - gMin) / (gMax - gMin));

[aLtpNorm, yLtpFit, errLtp] = fitLTP(ltpX, ltpY);
[aLtdNorm, yLtdFit, errLtd] = fitLTD(ltdX, ltdY);

maxNumLevelLTP = max(1, round(max(ltp(:,1)) - min(ltp(:,1))));
maxNumLevelLTD = max(1, round(max(ltd(:,1)) - min(ltd(:,1))));
sigmaCtoCNorm = mean([std(errLtp), std(errLtd)], "omitnan");
if ~isfinite(sigmaCtoCNorm)
    sigmaCtoCNorm = 0;
end

result = struct();
result.maxConductance = gMax;
result.minConductance = gMin;
result.maxNumLevelLTP = maxNumLevelLTP;
result.maxNumLevelLTD = maxNumLevelLTD;
result.paramALTPNorm = aLtpNorm;
result.paramALTDNorm = aLtdNorm;
result.paramALTP = aLtpNorm * maxNumLevelLTP;
result.paramALTD = aLtdNorm * maxNumLevelLTD;
result.sigmaCtoCNorm = sigmaCtoCNorm;
result.readVoltage = opts.ReadVoltage;
result.writeVoltageLTP = opts.WriteVoltageLTP;
result.writeVoltageLTD = opts.WriteVoltageLTD;
result.writePulseWidthLTP = opts.WritePulseWidthLTP;
result.writePulseWidthLTD = opts.WritePulseWidthLTD;

if opts.ShowFigure || opts.SaveFigure
    fig = figure("Name", "Fitted LTP/LTD curves");
    plot(ltpX, ltpY, "bo", "LineWidth", 1.2); hold on;
    plot(ltpX, yLtpFit, "b-", "LineWidth", 2);
    plot(ltdX, ltdY, "ro", "LineWidth", 1.2);
    plot(ltdX, yLtdFit, "r-", "LineWidth", 2);
    grid on;
    xlabel("Normalized pulse number");
    ylabel("Normalized conductance");
    legend("LTP data", "LTP fit", "LTD data", "LTD fit", "Location", "best");
    title("Device LTP/LTD fitting for MLP NeuroSim RealDevice");
    if opts.SaveFigure
        saveas(fig, fullfile(fileparts(mfilename("fullpath")), "fit_device_result.png"));
    end
    if ~opts.ShowFigure
        close(fig);
    end
end

writeReport(opts.ReportFile, result);

if opts.ApplyToCellCpp
    updateCellCpp(opts.CellCpp, result);
end

disp("Fit complete.");
disp(result);
end

function opts = parseNameValue(opts, varargin)
if mod(numel(varargin), 2) ~= 0
    error("Optional arguments must be name/value pairs.");
end
for i = 1:2:numel(varargin)
    name = char(varargin{i});
    if ~isfield(opts, name)
        error("Unknown option: %s", name);
    end
    opts.(name) = varargin{i+1};
end
end

function data = readCurve(file)
if ~isfile(file)
    error("Cannot find data file: %s", file);
end
raw = readmatrix(file);
if size(raw, 2) < 2
    error("Data file must contain at least two columns: pulse, conductance.");
end
data = raw(:, 1:2);
data = data(all(isfinite(data), 2), :);
if size(data, 1) < 3
    error("Need at least three valid data points in %s.", file);
end
data = sortrows(data, 1);
end

function x = normalizePulse(pulse)
span = max(pulse) - min(pulse);
if span <= 0
    error("Pulse numbers must span more than one value.");
end
x = (pulse - min(pulse)) / span;
end

function y = clamp01(y)
y = min(max(y, 0), 1);
end

function [aNorm, yFit, err] = fitLTP(x, y)
amin = 0.01;
amax = 100;
objective = @(z) mean((ltpModel(x, boundedPositive(z, amin, amax)) - y).^2);
z = fminsearch(objective, 0, optimset("Display", "off"));
aNorm = boundedPositive(z, amin, amax);
yFit = ltpModel(x, aNorm);
err = y - yFit;
end

function y = ltpModel(x, a)
b = 1 / (1 - exp(-1 / a));
y = b * (1 - exp(-x / a));
y = clamp01(y);
end

function [aNorm, yFit, err] = fitLTD(x, y)
amin = 0.01;
amax = 100;
objective = @(z) mean((ltdModel(x, -boundedPositive(z, amin, amax)) - y).^2);
z = fminsearch(objective, 0, optimset("Display", "off"));
aNorm = -boundedPositive(z, amin, amax);
yFit = ltdModel(x, aNorm);
err = y - yFit;
end

function a = boundedPositive(z, amin, amax)
a = amin + (amax - amin) / (1 + exp(-z));
end

function y = ltdModel(x, a)
state = 1 - x;
b = 1 / (1 - exp(-1 / a));
y = b * (1 - exp(-state / a));
y = clamp01(y);
end

function writeReport(reportFile, result)
fid = fopen(reportFile, "w");
if fid < 0
    error("Cannot write report: %s", reportFile);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "Fitted RealDevice parameters\n");
fprintf(fid, "maxConductance = %.12g\n", result.maxConductance);
fprintf(fid, "minConductance = %.12g\n", result.minConductance);
fprintf(fid, "maxNumLevelLTP = %d\n", result.maxNumLevelLTP);
fprintf(fid, "maxNumLevelLTD = %d\n", result.maxNumLevelLTD);
fprintf(fid, "paramALTP normalized = %.12g\n", result.paramALTPNorm);
fprintf(fid, "paramALTD normalized = %.12g\n", result.paramALTDNorm);
fprintf(fid, "paramALTP C++ = %.12g\n", result.paramALTP);
fprintf(fid, "paramALTD C++ = %.12g\n", result.paramALTD);
fprintf(fid, "sigmaCtoC normalized = %.12g\n", result.sigmaCtoCNorm);
fprintf(fid, "readVoltage = %.12g\n", result.readVoltage);
fprintf(fid, "writeVoltageLTP = %.12g\n", result.writeVoltageLTP);
fprintf(fid, "writeVoltageLTD = %.12g\n", result.writeVoltageLTD);
fprintf(fid, "writePulseWidthLTP = %.12g\n", result.writePulseWidthLTP);
fprintf(fid, "writePulseWidthLTD = %.12g\n", result.writePulseWidthLTD);
end

function updateCellCpp(cellCpp, result)
if ~isfile(cellCpp)
    error("Cannot find Cell.cpp: %s", cellCpp);
end
text = fileread(cellCpp);
backup = sprintf("%s.bak.%s", cellCpp, datestr(now, "yyyymmdd_HHMMSS"));
fid = fopen(backup, "w");
if fid < 0
    error("Cannot write backup: %s", backup);
end
fwrite(fid, text);
fclose(fid);

startToken = 'RealDevice::RealDevice(int x, int y) {';
endToken = 'double RealDevice::Read';
matchStart = strfind(text, startToken);
readStart = strfind(text, endToken);
if isempty(matchStart) || isempty(readStart)
    error("Could not locate RealDevice::RealDevice block in Cell.cpp.");
end
matchStart = matchStart(1);
readStart = readStart(find(readStart > matchStart, 1));
if isempty(readStart)
    error("Could not locate RealDevice::Read after RealDevice::RealDevice in Cell.cpp.");
end
block = text(matchStart:readStart-1);
openBrace = strfind(block, "{");
closeBrace = find(block == '}', 1, "last");
if isempty(openBrace) || isempty(closeBrace)
    error("Could not parse RealDevice::RealDevice braces in Cell.cpp.");
end

blockHead = block(1:openBrace(1));
body = block(openBrace(1)+1:closeBrace-1);
blockTail = block(closeBrace:end);
body = regexprep(body, ...
    "maxConductance\s*=\s*[-+0-9.eE]+;\s*// Maximum cell conductance \(S\)", ...
    sprintf("maxConductance = %.12g;\t\t// Maximum cell conductance (S)", result.maxConductance), ...
    "once");
body = regexprep(body, ...
    "minConductance\s*=\s*[-+0-9.eE]+;\s*// Minimum cell conductance \(S\)", ...
    sprintf("minConductance = %.12g;\t// Minimum cell conductance (S)", result.minConductance), ...
    "once");
body = regexprep(body, ...
    "readVoltage\s*=\s*[-+0-9.eE]+;\s*// On-chip read voltage \(Vr\) \(V\)", ...
    sprintf("readVoltage = %.12g;\t// On-chip read voltage (Vr) (V)", result.readVoltage), ...
    "once");
body = regexprep(body, ...
    "writeVoltageLTP\s*=\s*[-+0-9.eE]+;\s*// Write voltage \(V\) for LTP or weight increase", ...
    sprintf("writeVoltageLTP = %.12g;\t// Write voltage (V) for LTP or weight increase", result.writeVoltageLTP), ...
    "once");
body = regexprep(body, ...
    "writeVoltageLTD\s*=\s*[-+0-9.eE]+;\s*// Write voltage \(V\) for LTD or weight decrease", ...
    sprintf("writeVoltageLTD = %.12g;\t// Write voltage (V) for LTD or weight decrease", result.writeVoltageLTD), ...
    "once");
body = regexprep(body, ...
    "writePulseWidthLTP\s*=\s*[-+0-9.eE]+;\s*// Write pulse width \(s\) for LTP or weight increase", ...
    sprintf("writePulseWidthLTP = %.12g;\t// Write pulse width (s) for LTP or weight increase", result.writePulseWidthLTP), ...
    "once");
body = regexprep(body, ...
    "writePulseWidthLTD\s*=\s*[-+0-9.eE]+;\s*// Write pulse width \(s\) for LTD or weight decrease", ...
    sprintf("writePulseWidthLTD = %.12g;\t// Write pulse width (s) for LTD or weight decrease", result.writePulseWidthLTD), ...
    "once");
body = regexprep(body, ...
    "maxNumLevelLTP\s*=\s*\d+;\s*// Maximum number of conductance states during LTP or weight increase", ...
    sprintf("maxNumLevelLTP = %d;\t// Maximum number of conductance states during LTP or weight increase", result.maxNumLevelLTP), ...
    "once");
body = regexprep(body, ...
    "maxNumLevelLTD\s*=\s*\d+;\s*// Maximum number of conductance states during LTD or weight decrease", ...
    sprintf("maxNumLevelLTD = %d;\t// Maximum number of conductance states during LTD or weight decrease", result.maxNumLevelLTD), ...
    "once");
body = regexprep(body, ...
    "paramALTP\s*=\s*[^;\n]+;\s*// Parameter A for LTP nonlinearity[^\n]*", ...
    sprintf("paramALTP = %.12g;\t// Parameter A for LTP nonlinearity (fitted by MATLAB)", result.paramALTP), ...
    "once");
body = regexprep(body, ...
    "paramALTD\s*=\s*[^;\n]+;\s*// Parameter A for LTD nonlinearity[^\n]*", ...
    sprintf("paramALTD = %.12g;\t// Parameter A for LTD nonlinearity (fitted by MATLAB)", result.paramALTD), ...
    "once");
body = regexprep(body, ...
    "sigmaCtoC\s*=\s*[-+0-9.eE]+\s*\*\s*\(maxConductance\s*-\s*minConductance\);\s*// Sigma of cycle-to-cycle weight update vairation: defined as the percentage of conductance range", ...
    sprintf("sigmaCtoC = %.12g * (maxConductance - minConductance);\t// Sigma of cycle-to-cycle weight update vairation: defined as the percentage of conductance range", result.sigmaCtoCNorm), ...
    "once");

newBlock = [blockHead, body, blockTail];
text = [text(1:matchStart-1), newBlock, text(readStart:end)];

fid = fopen(cellCpp, "w");
if fid < 0
    error("Cannot write Cell.cpp: %s", cellCpp);
end
fwrite(fid, text);
fclose(fid);
fprintf("Updated %s\nBackup: %s\n", cellCpp, backup);
end

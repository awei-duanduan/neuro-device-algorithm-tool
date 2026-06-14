function fit_device_gui()
%FIT_DEVICE_GUI GUI for fitting device curves and updating RealDevice.

rootDir = char(fileparts(mfilename('fullpath')));
state = struct();
state.ltpFile = char(fullfile(rootDir, 'volatile_LTP_example.csv'));
state.ltdFile = char(fullfile(rootDir, 'volatile_LTD_example.csv'));
state.result = [];
state.runProcess = [];
state.runTimer = [];
state.runStartTime = [];
state.runLogPath = char(fullfile(rootDir, 'neurosim_run_log.txt'));
state.runKind = "cpp";
state.torchOutputDir = char(fullfile(rootDir, 'torch_results_gui'));
state.rcOutputDir = char(fullfile(rootDir, 'rc_results_gui_paper_multitask'));
state.rcResponseFile = char(fullfile(rootDir, 'reservoir_device_response_pndi_05s.csv'));
state.totalEpochs = readTotalEpochs();
state.seedPath = char(fullfile(rootDir, 'run_seed.txt'));

fig = figure( ...
    "Name", "MLP NeuroSim Device Fitting", ...
    "NumberTitle", "off", ...
    "MenuBar", "none", ...
    "ToolBar", "none", ...
    "Color", [0.96 0.96 0.96], ...
    "Position", [120 40 1180 800], ...
    "Resize", "on");
set(fig, "CloseRequestFcn", @(~,~) closeGui());

tabs = uitabgroup(fig, "Units", "normalized", "Position", [0 0 1 1]);
mlpTab = uitab(tabs, "Title", "MLP / NeuroSim");
rcTab = uitab(tabs, "Title", "Reservoir Computing");

left = uipanel(mlpTab, "Title", "Input and Options", "FontWeight", "bold", ...
    "Units", "normalized", "Position", [0.02 0.03 0.34 0.94]);
right = uipanel(mlpTab, "Title", "Fit Preview", "FontWeight", "bold", ...
    "Units", "normalized", "Position", [0.38 0.03 0.60 0.94]);

ax = axes(right, "Units", "normalized", "Position", [0.09 0.34 0.86 0.54]);
grid(ax, "on");
xlabel(ax, "Normalized pulse number");
ylabel(ax, "Normalized conductance");
title(ax, "Select data and click Fit Preview");

uicontrol(right, "Style", "text", "String", "Dataset", ...
    "HorizontalAlignment", "left", "FontWeight", "bold", ...
    "BackgroundColor", get(right, "BackgroundColor"), ...
    "Units", "normalized", "Position", [0.05 0.945 0.14 0.03]);
datasetPopup = uicontrol(right, "Style", "popupmenu", ...
    "String", {"MNIST digits", "Fashion-MNIST fashion"}, ...
    "Value", 1, ...
    "Units", "normalized", "Position", [0.17 0.945 0.36 0.035]);

seedText = uicontrol(right, "Style", "text", ...
    "String", "", ...
    "HorizontalAlignment", "left", ...
    "BackgroundColor", get(right, "BackgroundColor"), ...
    "Units", "normalized", "Position", [0.58 0.945 0.18 0.030]);

uicontrol(right, "Style", "pushbutton", "String", "New Random Init", ...
    "Units", "normalized", "Position", [0.77 0.943 0.18 0.035], ...
    "Callback", @(~,~) newRandomInit());

progressText = uicontrol(right, "Style", "text", ...
    "String", "Run progress: idle", ...
    "HorizontalAlignment", "left", ...
    "BackgroundColor", get(right, "BackgroundColor"), ...
    "Units", "normalized", "Position", [0.05 0.275 0.90 0.03]);
progressAx = axes(right, "Units", "normalized", "Position", [0.05 0.245 0.90 0.025]);
axis(progressAx, [0 1 0 1]);
set(progressAx, "XTick", [], "YTick", [], "Box", "on", "Color", [0.90 0.90 0.90]);
progressPatch = patch(progressAx, [0 0 0 0], [0 0 1 1], [0.10 0.45 0.85], ...
    "EdgeColor", "none");

statusBox = uicontrol(right, "Style", "edit", "Max", 12, "Min", 0, ...
    "Enable", "inactive", "HorizontalAlignment", "left", ...
    "Units", "normalized", "Position", [0.05 0.04 0.90 0.18], ...
    "String", "Ready.");

row = 0.91;
addLabel(left, "LTP data file", row);
ltpEdit = addEdit(left, state.ltpFile, row - 0.040);
uicontrol(left, "Style", "pushbutton", "String", "Browse", ...
    "Units", "normalized", "Position", [0.72 row-0.047 0.22 0.038], ...
    "Callback", @(~,~) browseFile("ltp"));

row = row - 0.115;
addLabel(left, "LTD data file", row);
ltdEdit = addEdit(left, state.ltdFile, row - 0.040);
uicontrol(left, "Style", "pushbutton", "String", "Browse", ...
    "Units", "normalized", "Position", [0.72 row-0.047 0.22 0.038], ...
    "Callback", @(~,~) browseFile("ltd"));

row = row - 0.105;
addLabel(left, "Read voltage (V)", row);
readVoltageEdit = addEdit(left, "0.5", row - 0.040);

row = row - 0.080;
addLabel(left, "LTP write voltage (V)", row);
writeVoltageLTPEdit = addEdit(left, "3.2", row - 0.040);

row = row - 0.080;
addLabel(left, "LTD write voltage (V)", row);
writeVoltageLTDEdit = addEdit(left, "2.8", row - 0.040);

row = row - 0.080;
addLabel(left, "LTP pulse width (s)", row);
writePulseWidthLTPEdit = addEdit(left, "300e-6", row - 0.040);

row = row - 0.080;
addLabel(left, "LTD pulse width (s)", row);
writePulseWidthLTDEdit = addEdit(left, "300e-6", row - 0.040);

row = row - 0.080;
addLabel(left, "Hidden neurons (e.g. 30, 64, 100)", row);
hiddenNeuronsEdit = addEdit(left, num2str(readParamValue('nHide', 100)), row - 0.045);
set(hiddenNeuronsEdit, "Position", [0.06 row-0.052 0.62 0.045]);

row = row - 0.070;
addLabel(left, "Epochs", row);
epochsEdit = addEdit(left, num2str(readParamValue('totalNumEpochs', 125)), row - 0.042);
set(epochsEdit, "Position", [0.06 row-0.048 0.62 0.042]);

applyCheck = uicontrol(left, "Style", "checkbox", ...
    "String", "Update Cell.cpp after fitting", ...
    "Value", 1, "Units", "normalized", ...
    "Position", [0.06 0.132 0.82 0.028]);

uicontrol(left, "Style", "pushbutton", "String", "Fit Preview", ...
    "FontWeight", "bold", "Units", "normalized", ...
    "Position", [0.06 0.099 0.40 0.028], ...
    "Callback", @(~,~) runFit(false));

uicontrol(left, "Style", "pushbutton", "String", "Fit and Update", ...
    "FontWeight", "bold", "Units", "normalized", ...
    "Position", [0.54 0.099 0.40 0.028], ...
    "Callback", @(~,~) runFit(true));

uicontrol(left, "Style", "pushbutton", "String", "Run Neural Network", ...
    "FontWeight", "bold", "Units", "normalized", ...
    "Position", [0.06 0.067 0.40 0.028], ...
    "Callback", @(~,~) runNeuralNetwork());

uicontrol(left, "Style", "pushbutton", "String", "Run PyTorch CUDA", ...
    "FontWeight", "bold", "Units", "normalized", ...
    "Position", [0.54 0.067 0.40 0.028], ...
    "Callback", @(~,~) runPyTorchCuda());

uicontrol(left, "Style", "pushbutton", "String", "Stop Run", ...
    "Units", "normalized", ...
    "Position", [0.06 0.036 0.40 0.027], ...
    "Callback", @(~,~) stopRunProcess(true));

uicontrol(left, "Style", "pushbutton", "String", "Open Report", ...
    "Units", "normalized", "Position", [0.54 0.036 0.40 0.027], ...
    "Callback", @(~,~) openReport());

uicontrol(left, "Style", "pushbutton", "String", "Open Project Folder", ...
    "Units", "normalized", "Position", [0.06 0.006 0.88 0.025], ...
    "Callback", @(~,~) winopen(rootDir));

rcLeft = uipanel(rcTab, "Title", "Reservoir Options", "FontWeight", "bold", ...
    "Units", "normalized", "Position", [0.02 0.03 0.34 0.94]);
rcRight = uipanel(rcTab, "Title", "Reservoir Results", "FontWeight", "bold", ...
    "Units", "normalized", "Position", [0.38 0.03 0.60 0.94]);

rcAx = axes(rcRight, "Units", "normalized", "Position", [0.09 0.39 0.86 0.49]);
grid(rcAx, "on");
xlabel(rcAx, "Epoch");
ylabel(rcAx, "Accuracy % / Cross-Entropy");
title(rcAx, "Run Reservoir Computing");

rcSeedText = uicontrol(rcRight, "Style", "text", ...
    "String", "", ...
    "HorizontalAlignment", "left", ...
    "BackgroundColor", get(rcRight, "BackgroundColor"), ...
    "Units", "normalized", "Position", [0.05 0.925 0.36 0.030]);

uicontrol(rcRight, "Style", "pushbutton", "String", "New Random Init", ...
    "Units", "normalized", "Position", [0.43 0.922 0.18 0.035], ...
    "Callback", @(~,~) newRandomInit());

uicontrol(rcRight, "Style", "pushbutton", "String", "Paper Defaults", ...
    "Units", "normalized", "Position", [0.63 0.922 0.18 0.035], ...
    "Callback", @(~,~) applyRcPaperDefaults());

rcProgressText = uicontrol(rcRight, "Style", "text", ...
    "String", "Run progress: idle", ...
    "HorizontalAlignment", "left", ...
    "BackgroundColor", get(rcRight, "BackgroundColor"), ...
    "Units", "normalized", "Position", [0.05 0.285 0.90 0.03]);
rcProgressAx = axes(rcRight, "Units", "normalized", "Position", [0.05 0.255 0.90 0.025]);
axis(rcProgressAx, [0 1 0 1]);
set(rcProgressAx, "XTick", [], "YTick", [], "Box", "on", "Color", [0.90 0.90 0.90]);
rcProgressPatch = patch(rcProgressAx, [0 0 0 0], [0 0 1 1], [0.12 0.55 0.35], ...
    "EdgeColor", "none");

rcStatusBox = uicontrol(rcRight, "Style", "edit", "Max", 12, "Min", 0, ...
    "Enable", "inactive", "HorizontalAlignment", "left", ...
    "Units", "normalized", "Position", [0.05 0.04 0.90 0.18], ...
    "String", sprintf([ ...
        'Ready.\n', ...
        'Reservoir flow: binarized image -> 4-bit pulse codes -> device response table -> linear readout.\n', ...
        'Default response file is an approximate Pei 2025 Fig. 7d table.']));

rcRow = 0.90;
addLabel(rcLeft, "Dataset", rcRow);
rcDatasetPopup = uicontrol(rcLeft, "Style", "popupmenu", ...
    "String", {"Paper multitask (MNIST + EMNIST + Fashion)", "MNIST digits", "Fashion-MNIST fashion"}, ...
    "Value", 1, ...
    "Units", "normalized", "Position", [0.06 rcRow-0.045 0.62 0.045]);

rcRow = rcRow - 0.105;
addLabel(rcLeft, "Device response CSV", rcRow);
rcResponseEdit = addEdit(rcLeft, state.rcResponseFile, rcRow - 0.045);
set(rcResponseEdit, "Position", [0.06 rcRow-0.052 0.62 0.045]);
uicontrol(rcLeft, "Style", "pushbutton", "String", "Browse", ...
    "Units", "normalized", "Position", [0.72 rcRow-0.052 0.22 0.045], ...
    "Callback", @(~,~) browseRcResponse());

rcRow = rcRow - 0.105;
addLabel(rcLeft, "Epochs", rcRow);
rcEpochsEdit = addEdit(rcLeft, "30", rcRow - 0.045);
set(rcEpochsEdit, "Position", [0.06 rcRow-0.052 0.62 0.045]);

rcRow = rcRow - 0.090;
addLabel(rcLeft, "Train samples (0 = full)", rcRow);
rcTrainLimitEdit = addEdit(rcLeft, "10000", rcRow - 0.045);
set(rcTrainLimitEdit, "Position", [0.06 rcRow-0.052 0.62 0.045]);

rcRow = rcRow - 0.090;
addLabel(rcLeft, "Test samples (0 = full)", rcRow);
rcTestLimitEdit = addEdit(rcLeft, "2000", rcRow - 0.045);
set(rcTestLimitEdit, "Position", [0.06 rcRow-0.052 0.62 0.045]);

rcRow = rcRow - 0.090;
addLabel(rcLeft, "Batch size", rcRow);
rcBatchEdit = addEdit(rcLeft, "512", rcRow - 0.045);
set(rcBatchEdit, "Position", [0.06 rcRow-0.052 0.62 0.045]);

uicontrol(rcLeft, "Style", "pushbutton", "String", "Run Reservoir Computing", ...
    "FontWeight", "bold", "Units", "normalized", ...
    "Position", [0.06 0.170 0.88 0.040], ...
    "Callback", @(~,~) runReservoirComputing());

uicontrol(rcLeft, "Style", "pushbutton", "String", "Stop Run", ...
    "Units", "normalized", "Position", [0.06 0.118 0.40 0.035], ...
    "Callback", @(~,~) stopRunProcess(true));

uicontrol(rcLeft, "Style", "pushbutton", "String", "Open RC Results", ...
    "Units", "normalized", "Position", [0.54 0.118 0.40 0.035], ...
    "Callback", @(~,~) openRcResults());

uicontrol(rcLeft, "Style", "pushbutton", "String", "Open Project Folder", ...
    "Units", "normalized", "Position", [0.06 0.067 0.88 0.035], ...
    "Callback", @(~,~) winopen(rootDir));

updateSeedText();


    function addLabel(parent, text, y)
        uicontrol(parent, "Style", "text", "String", text, ...
            "HorizontalAlignment", "left", "FontWeight", "bold", ...
            "BackgroundColor", get(parent, "BackgroundColor"), ...
            "Units", "normalized", "Position", [0.06 y 0.88 0.035]);
    end

    function h = addEdit(parent, text, y)
        h = uicontrol(parent, "Style", "edit", "String", text, ...
            "HorizontalAlignment", "left", ...
            "Units", "normalized", "Position", [0.06 y 0.62 0.04]);
    end

    function newRandomInit()
        rng('shuffle');
        weightSeed = randi([1, 2147480000]);
        trainSeed = randi([1, 2147480000]);
        writeSeedFile(weightSeed, trainSeed);
        updateSeedText();
        set(statusBox, "String", sprintf([ ...
            'New random initialization selected.\n', ...
            'Weight seed = %d\n', ...
            'Training seed = %d\n', ...
            'Click Run Neural Network to use it.'], weightSeed, trainSeed));
        set(rcStatusBox, "String", sprintf([ ...
            'New random initialization selected.\n', ...
            'Seed = %d\n', ...
            'Click Run Reservoir Computing to use it.'], trainSeed));
    end

    function writeSeedFile(weightSeed, trainSeed)
        fid = fopen(state.seedPath, 'w');
        if fid < 0
            error("Could not write random seed file: %s", state.seedPath);
        end
        cleaner = onCleanup(@() fclose(fid));
        fprintf(fid, '%d %d\n', weightSeed, trainSeed);
        clear cleaner;
    end

    function [weightSeed, trainSeed] = readSeedFile()
        weightSeed = 2;
        trainSeed = 0;
        if isfile(state.seedPath)
            values = sscanf(fileread(state.seedPath), '%d');
            if numel(values) >= 1
                weightSeed = values(1);
            end
            if numel(values) >= 2
                trainSeed = values(2);
            end
        end
    end

    function updateSeedText()
        [weightSeed, trainSeed] = readSeedFile();
        set(seedText, "String", sprintf('Seed W=%d, T=%d', weightSeed, trainSeed));
        if exist('rcSeedText', 'var') && isgraphics(rcSeedText)
            set(rcSeedText, "String", sprintf('Seed = %d', trainSeed));
        end
    end

    function applyRcPaperDefaults()
        set(rcDatasetPopup, "Value", 1);
        set(rcResponseEdit, "String", char(fullfile(rootDir, 'reservoir_device_response_pndi_05s.csv')));
        set(rcEpochsEdit, "String", "100");
        set(rcTrainLimitEdit, "String", "0");
        set(rcTestLimitEdit, "String", "0");
        set(rcBatchEdit, "String", "2048");
        state.rcResponseFile = char(fullfile(rootDir, 'reservoir_device_response_pndi_05s.csv'));
        set(rcStatusBox, "String", sprintf([ ...
            'Paper defaults applied.\n', ...
            'Epochs = 100 per independent task\n', ...
            'Batch size = 2048\n', ...
            'Train/test samples = full available dataset after task filtering.\n', ...
            'Device response = p-NDI 5-bit / 32 states.']));
    end

    function browseFile(which)
        [file, path] = uigetfile( ...
            {"*.csv;*.txt;*.xlsx;*.xls", "Data files (*.csv, *.txt, *.xlsx, *.xls)"; "*.*", "All files"}, ...
            "Select device curve data", rootDir);
        if isequal(file, 0)
            return;
        end
        selected = fullfile(path, file);
        if strcmp(which, "ltp")
            state.ltpFile = selected;
            set(ltpEdit, "String", selected);
        else
            state.ltdFile = selected;
            set(ltdEdit, "String", selected);
        end
    end

    function browseRcResponse()
        [file, path] = uigetfile( ...
            {"*.csv", "CSV files (*.csv)"; "*.*", "All files"}, ...
            "Select reservoir device response table", rootDir);
        if isequal(file, 0)
            return;
        end
        state.rcResponseFile = fullfile(path, file);
        set(rcResponseEdit, "String", state.rcResponseFile);
    end

    function runReservoirComputing()
        try
            rcScript = char(fullfile(rootDir, 'rc_reservoir_train.py'));
            pythonExe = char(fullfile(rootDir, '.venv_torch', 'Scripts', 'python.exe'));
            if ~isfile(rcScript)
                error("Reservoir script not found: %s", rcScript);
            end
            if ~isfile(pythonExe)
                error("PyTorch virtual environment not found: %s", pythonExe);
            end

            responseFile = char(strtrim(get(rcResponseEdit, "String")));
            if ~isfile(responseFile)
                error("Device response CSV not found: %s", responseFile);
            end
            epochs = readPositiveInteger(rcEpochsEdit, "Epochs");
            trainLimit = readNonnegativeInteger(rcTrainLimitEdit, "Train samples");
            testLimit = readNonnegativeInteger(rcTestLimitEdit, "Test samples");
            batchSize = readPositiveInteger(rcBatchEdit, "Batch size");
            datasetKey = getSelectedRcDatasetKey();
            [~, trainSeed] = readSeedFile();

            if strcmp(datasetKey, 'paper')
                state.totalEpochs = epochs * 3;
            else
                state.totalEpochs = epochs;
            end
            state.rcResponseFile = responseFile;
            outputDir = char(fullfile(rootDir, sprintf('rc_results_gui_%s', datasetKey)));
            logPath = char(fullfile(rootDir, 'rc_run_log.txt'));
            state.rcOutputDir = outputDir;
            state.runKind = "rc";

            set(rcStatusBox, "String", sprintf([ ...
                'Starting reservoir computing in background...\n', ...
                'Dataset: %s\n', ...
                'Reservoir feature: paper mode uses 5-bit/32-state p-NDI response and 3 readouts\n', ...
                'Readout: MNIST 10, EMNIST L/M/S 3, Fashion 5\n', ...
                'Epochs: %d\n', ...
                'Train samples: %d, test samples: %d\n', ...
                'Response CSV: %s\n', ...
                'Output: %s\n', ...
                'Log: %s'], datasetKey, epochs, trainLimit, testLimit, responseFile, outputDir, logPath));
            setRunProgress(0.02, "Preparing reservoir computing run...");
            drawnow;

            batchFile = writeRcRunBatch(datasetKey, responseFile, epochs, trainLimit, testLimit, batchSize, trainSeed, outputDir, logPath);
            startRunProcess(batchFile, logPath, "rc");
        catch ME
            set(rcStatusBox, "String", ['Error: ', char(ME.message)]);
            setRunProgress(0, "Reservoir run failed before start.");
            errordlg(ME.message, "Reservoir computing run failed");
        end
    end

    function key = getSelectedRcDatasetKey()
        value = get(rcDatasetPopup, "Value");
        if value == 1
            key = 'paper';
        elseif value == 3
            key = 'fashion';
        else
            key = 'mnist';
        end
    end

    function value = readPositiveInteger(editHandle, label)
        value = str2double(get(editHandle, "String"));
        if ~isfinite(value) || value <= 0 || abs(value - round(value)) > 0
            error("%s must be a positive integer.", label);
        end
        value = round(value);
    end

    function value = readNonnegativeInteger(editHandle, label)
        value = str2double(get(editHandle, "String"));
        if ~isfinite(value) || value < 0 || abs(value - round(value)) > 0
            error("%s must be a nonnegative integer.", label);
        end
        value = round(value);
    end

    function value = readNumber(editHandle, label)
        value = str2double(get(editHandle, "String"));
        if ~isfinite(value)
            error("%s must be a valid number.", label);
        end
    end

    function ok = runFit(forceApply)
        ok = false;
        try
            applyNow = forceApply && logical(get(applyCheck, "Value"));
            set(statusBox, "String", "Running MATLAB fit...");
            drawnow;

            result = fit_device_to_realsim( ...
                resolveDataPath(get(ltpEdit, "String")), resolveDataPath(get(ltdEdit, "String")), ...
                "ReadVoltage", readNumber(readVoltageEdit, "Read voltage"), ...
                "WriteVoltageLTP", readNumber(writeVoltageLTPEdit, "LTP write voltage"), ...
                "WriteVoltageLTD", readNumber(writeVoltageLTDEdit, "LTD write voltage"), ...
                "WritePulseWidthLTP", readNumber(writePulseWidthLTPEdit, "LTP pulse width"), ...
                "WritePulseWidthLTD", readNumber(writePulseWidthLTDEdit, "LTD pulse width"), ...
                "ApplyToCellCpp", applyNow, ...
                "ShowFigure", false, ...
                "SaveFigure", true);

            state.result = result;
            drawPreview(ax, resolveDataPath(get(ltpEdit, "String")), resolveDataPath(get(ltdEdit, "String")));
            message = sprintf([ ...
                'Fit complete.\n', ...
                'Gmax = %.4g S\nGmin = %.4g S\n', ...
                'LTP levels = %d, LTD levels = %d\n', ...
                'A_LTP = %.4g * maxNumLevelLTP\n', ...
                'A_LTD = %.4g * maxNumLevelLTD\n', ...
                'sigmaCtoC = %.4g * conductance range\n'], ...
                result.maxConductance, result.minConductance, ...
                result.maxNumLevelLTP, result.maxNumLevelLTD, ...
                result.paramALTPNorm, result.paramALTDNorm, ...
                result.sigmaCtoCNorm);
            if applyNow
                message = [message, sprintf('\nCell.cpp updated. Backup was created.')];
            else
                message = [message, sprintf('\nPreview only. Cell.cpp was not changed.')];
            end
            set(statusBox, "String", message);
            ok = true;
        catch ME
            set(statusBox, "String", ['Error: ', char(ME.message)]);
            errordlg(ME.message, "Fit failed");
        end
    end

    function runNeuralNetwork()
        oldApplyValue = get(applyCheck, "Value");
        try
            set(applyCheck, "Value", 1);
            fitOk = runFit(true);
            if ~fitOk
                set(applyCheck, "Value", oldApplyValue);
                return;
            end

            logPath = char(fullfile(rootDir, 'neurosim_run_log.txt'));
            set(statusBox, "String", sprintf([ ...
                'Cell.cpp updated.\n', ...
                'Starting C++ neural network in background...\n', ...
                'This may take several minutes.\n', ...
                'Log: %s'], logPath));
            setRunProgress(0.02, "Preparing build...");
            drawnow;

            msysPath = 'C:\msys64\ucrt64\bin;C:\msys64\usr\bin;%PATH%';
            [nInput, nHide, imageSize, epochs] = readNetworkShape();
            applyNetworkShape(nInput, nHide, epochs);
            applyCppPostMappingMode();
            state.totalEpochs = epochs;
            datasetName = prepareSelectedDataset(imageSize);
            [weightSeed, trainSeed] = readSeedFile();
            batchFile = writeRunBatch(msysPath, logPath);
            set(statusBox, "String", sprintf([ ...
                'Cell.cpp updated.\n', ...
                'Dataset: %s\n', ...
                'Network: %d-%d-10\n', ...
                'Epochs: %d\n', ...
                'Mode: software training + RealDevice post mapping test\n', ...
                'Seed W=%d, T=%d\n', ...
                'Starting C++ neural network in background...\n', ...
                'Log: %s'], datasetName, nInput, nHide, epochs, weightSeed, trainSeed, logPath));
            startRunProcess(batchFile, logPath);
        catch ME
            set(statusBox, "String", ['Error: ', char(ME.message)]);
            setRunProgress(0, "Run failed before start.");
            errordlg(ME.message, "Neural network run failed");
        end
        set(applyCheck, "Value", oldApplyValue);
    end

    function runPyTorchCuda()
        try
            torchScript = char(fullfile(rootDir, 'torch_neurosim_train.py'));
            pythonExe = char(fullfile(rootDir, '.venv_torch', 'Scripts', 'python.exe'));
            if ~isfile(torchScript)
                error("PyTorch script not found: %s", torchScript);
            end
            if ~isfile(pythonExe)
                error("PyTorch virtual environment not found: %s", pythonExe);
            end

            [~, nHide, ~, epochs] = readNetworkShape();
            state.totalEpochs = epochs;
            [~, trainSeed] = readSeedFile();
            datasetKey = getSelectedTorchDatasetKey();
            outputDir = char(fullfile(rootDir, sprintf('torch_results_gui_%s', datasetKey)));
            logPath = char(fullfile(rootDir, 'torch_cuda_run_log.txt'));
            state.torchOutputDir = outputDir;

            set(statusBox, "String", sprintf([ ...
                'Starting PyTorch CUDA neural network in background...\n', ...
                'Dataset: %s\n', ...
                'Network: 784-%d-10\n', ...
                'Epochs: %d\n', ...
                'Seed: %d\n', ...
                'Output: %s\n', ...
                'Log: %s'], datasetKey, nHide, epochs, trainSeed, outputDir, logPath));
            setRunProgress(0.02, "Preparing PyTorch CUDA run...");
            drawnow;

            batchFile = writeTorchRunBatch(datasetKey, nHide, epochs, trainSeed, outputDir, logPath);
            startRunProcess(batchFile, logPath, "torch");
        catch ME
            set(statusBox, "String", ['Error: ', char(ME.message)]);
            setRunProgress(0, "PyTorch CUDA run failed before start.");
            errordlg(ME.message, "PyTorch CUDA run failed");
        end
    end

    function batchFile = writeRunBatch(msysPath, logPath)
        batchFile = char(fullfile(rootDir, 'run_neurosim_gui.bat'));
        fid = fopen(batchFile, 'w');
        if fid < 0
            error("Could not create run batch file: %s", batchFile);
        end
        cleaner = onCleanup(@() fclose(fid));
        fprintf(fid, '@echo off\r\n');
        fprintf(fid, 'set "PATH=%s"\r\n', msysPath);
        fprintf(fid, 'cd /d "%%~dp0"\r\n');
        fprintf(fid, 'make clean > "%s" 2>&1\r\n', logPath);
        fprintf(fid, 'if errorlevel 1 exit /b %%errorlevel%%\r\n');
        fprintf(fid, 'make >> "%s" 2>&1\r\n', logPath);
        fprintf(fid, 'if errorlevel 1 exit /b %%errorlevel%%\r\n');
        fprintf(fid, 'main.exe >> "%s" 2>&1\r\n', logPath);
        fprintf(fid, 'exit /b %%errorlevel%%\r\n');
        clear cleaner;
    end

    function batchFile = writeTorchRunBatch(datasetKey, nHide, epochs, seed, outputDir, logPath)
        batchFile = char(fullfile(rootDir, 'run_torch_cuda_gui.bat'));
        fid = fopen(batchFile, 'w');
        if fid < 0
            error("Could not create PyTorch run batch file: %s", batchFile);
        end
        [~, outputName, outputExt] = fileparts(outputDir);
        outputArg = [outputName, outputExt];
        [~, logName, logExt] = fileparts(logPath);
        logArg = [logName, logExt];
        pythonArg = char(fullfile('.venv_torch', 'Scripts', 'python.exe'));
        scriptArg = 'torch_neurosim_train.py';
        cleaner = onCleanup(@() fclose(fid));
        fprintf(fid, '@echo off\r\n');
        fprintf(fid, 'cd /d "%%~dp0"\r\n');
        fprintf(fid, 'if exist "%s" rmdir /s /q "%s"\r\n', outputArg, outputArg);
        fprintf(fid, '"%s" -u "%s" --dataset %s --hidden %d --epochs %d --batch-size 1024 --seed %d --optimizer adam --lr 0.001 --mapping-mode post --output-dir "%s" > "%s" 2>&1\r\n', ...
            pythonArg, scriptArg, datasetKey, nHide, epochs, seed, outputArg, logArg);
        fprintf(fid, 'exit /b %%errorlevel%%\r\n');
        clear cleaner;
    end

    function batchFile = writeRcRunBatch(datasetKey, responseFile, epochs, trainLimit, testLimit, batchSize, seed, outputDir, logPath)
        batchFile = char(fullfile(rootDir, 'run_reservoir_gui.bat'));
        fid = fopen(batchFile, 'w');
        if fid < 0
            error("Could not create reservoir run batch file: %s", batchFile);
        end
        [~, outputName, outputExt] = fileparts(outputDir);
        outputArg = [outputName, outputExt];
        [~, logName, logExt] = fileparts(logPath);
        logArg = [logName, logExt];
        pythonArg = char(fullfile('.venv_torch', 'Scripts', 'python.exe'));
        scriptArg = 'rc_reservoir_train.py';
        cleaner = onCleanup(@() fclose(fid));
        fprintf(fid, '@echo off\r\n');
        fprintf(fid, 'cd /d "%%~dp0"\r\n');
        fprintf(fid, 'if exist "%s" rmdir /s /q "%s"\r\n', outputArg, outputArg);
        if strcmp(datasetKey, 'paper')
            fprintf(fid, '"%s" -u "%s" --task paper-multitask --dataset mnist --device-response "%s" --epochs %d --batch-size %d --train-limit %d --test-limit %d --seed %d --output-dir "%s" > "%s" 2>&1\r\n', ...
                pythonArg, scriptArg, responseFile, epochs, batchSize, trainLimit, testLimit, seed, outputArg, logArg);
        else
            fprintf(fid, '"%s" -u "%s" --task single --dataset %s --device-response "%s" --epochs %d --batch-size %d --train-limit %d --test-limit %d --seed %d --output-dir "%s" > "%s" 2>&1\r\n', ...
                pythonArg, scriptArg, datasetKey, responseFile, epochs, batchSize, trainLimit, testLimit, seed, outputArg, logArg);
        end
        fprintf(fid, 'exit /b %%errorlevel%%\r\n');
        clear cleaner;
    end

    function startRunProcess(batchFile, logPath, runKind)
        if nargin < 3
            runKind = "cpp";
        end
        stopRunTimer();
        if isfile(logPath)
            delete(logPath);
        end

        psi = System.Diagnostics.ProcessStartInfo();
        psi.FileName = System.String('cmd.exe');
        [~, batchName, batchExt] = fileparts(batchFile);
        psi.Arguments = System.String(['/c "', batchName, batchExt, '"']);
        psi.WorkingDirectory = System.String(rootDir);
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;

        proc = System.Diagnostics.Process();
        proc.StartInfo = psi;
        started = proc.Start();
        if ~started
            error("Could not start C++ run process.");
        end

        state.runProcess = proc;
        state.runStartTime = tic;
        state.runLogPath = logPath;
        state.runKind = string(runKind);
        state.runTimer = timer( ...
            "ExecutionMode", "fixedSpacing", ...
            "Period", 2, ...
            "TimerFcn", @(~,~) pollRunProgress());
        start(state.runTimer);
        pollRunProgress();
    end

    function pollRunProgress()
        try
            elapsed = toc(state.runStartTime);
            logText = readLogText(state.runLogPath);
            latestEpoch = parseLatestEpoch(logText);
            if latestEpoch > 0
                fraction = min(0.05 + 0.95 * latestEpoch / max(state.totalEpochs, 1), 0.995);
                etaText = estimateRemaining(elapsed, fraction);
                label = sprintf('Running epoch %d/%d | elapsed %s | remaining %s', ...
                    latestEpoch, state.totalEpochs, formatDuration(elapsed), etaText);
            elseif state.runKind == "cpp" && (contains(logText, 'g++') || contains(logText, 'make'))
                fraction = 0.03;
                label = sprintf('Compiling C++ | elapsed %s | remaining estimating...', formatDuration(elapsed));
            elseif state.runKind == "torch" && (contains(logText, 'Device:') || contains(logText, 'Downloading'))
                fraction = 0.03;
                label = sprintf('Starting PyTorch CUDA | elapsed %s | remaining estimating...', formatDuration(elapsed));
            elseif state.runKind == "rc" && (contains(logText, 'RC Device:') || contains(logText, 'Downloading'))
                fraction = 0.03;
                label = sprintf('Starting reservoir computing | elapsed %s | remaining estimating...', formatDuration(elapsed));
            else
                fraction = 0.01;
                if state.runKind == "torch"
                    runName = 'PyTorch CUDA';
                elseif state.runKind == "rc"
                    runName = 'Reservoir Computing';
                else
                    runName = upper(char(state.runKind));
                end
                label = sprintf('Starting %s run | elapsed %s', runName, formatDuration(elapsed));
            end
            setRunProgress(fraction, label);

            if ~isempty(state.runProcess) && state.runProcess.HasExited
                exitCode = state.runProcess.ExitCode;
                stopRunTimer();
                if exitCode == 0
                    setRunProgress(1, sprintf('Complete | elapsed %s', formatDuration(elapsed)));
                    if state.runKind == "torch"
                        showTorchResults(elapsed);
                    elseif state.runKind == "rc"
                        showRcResults(elapsed);
                    else
                        [lossCsv, lossPng] = exportLossResults(state.runLogPath);
                        [confCountCsv, confPercentCsv, confPng, finalAccuracy] = exportConfusionResults(state.runLogPath);
                        exportAccuracyResults(state.runLogPath);
                        plotLossResults(lossCsv);
                        set(statusBox, "String", sprintf([ ...
                            'Neural network run complete.\n', ...
                            'Final accuracy = %.2f%%\n', ...
                            'Loss data: %s\n', ...
                            'Loss chart: %s\n', ...
                            'Confusion counts: %s\n', ...
                            'Confusion percent: %s\n', ...
                            'Confusion chart: %s\n', ...
                            'Log: %s'], finalAccuracy, ...
                            lossCsv, lossPng, confCountCsv, confPercentCsv, confPng, state.runLogPath));
                        if isfile(lossPng)
                            winopen(lossPng);
                        end
                        if isfile(confPng)
                            winopen(confPng);
                        end
                    end
                else
                    setRunProgress(fraction, sprintf('Failed | elapsed %s', formatDuration(elapsed)));
                    setActiveStatus(sprintf('%s run failed with exit code %d.\nLog: %s', upper(char(state.runKind)), exitCode, state.runLogPath));
                    errordlg(sprintf('%s run failed. See log:\n%s', upper(char(state.runKind)), state.runLogPath), "Neural network run failed");
                end
            end
            drawnow;
        catch ME
            stopRunTimer();
            setActiveStatus(['Error: ', char(ME.message)]);
            setRunProgress(0, "Progress monitor failed.");
        end
    end

    function showTorchResults(elapsed)
        summaryPath = char(fullfile(state.torchOutputDir, 'summary.json'));
        lossCsv = char(fullfile(state.torchOutputDir, 'loss_history.csv'));
        lossPng = char(fullfile(state.torchOutputDir, 'loss_curve.png'));
        accuracyCsv = char(fullfile(state.torchOutputDir, 'accuracy_history.csv'));
        accuracyPng = char(fullfile(state.torchOutputDir, 'accuracy_curve.png'));
        confCountCsv = char(fullfile(state.torchOutputDir, 'confusion_matrix_counts.csv'));
        confPercentCsv = char(fullfile(state.torchOutputDir, 'confusion_matrix_percent.csv'));
        confPng = char(fullfile(state.torchOutputDir, 'confusion_matrix.png'));
        if ~isfile(summaryPath)
            error("PyTorch summary file was not created: %s", summaryPath);
        end
        summary = jsondecode(fileread(summaryPath));
        finalAccuracy = summary.final_test_accuracy_percent;
        if isfield(summary, 'pre_mapping_test_accuracy_percent')
            preAccuracyText = sprintf('Pre-mapping accuracy = %.2f%%\n', summary.pre_mapping_test_accuracy_percent);
        else
            preAccuracyText = '';
        end
        plotLossResults(lossCsv);
        set(statusBox, "String", sprintf([ ...
            'PyTorch CUDA run complete.\n', ...
            'GPU = %s\n', ...
            'Network = %s\n', ...
            '%s', ...
            'Final accuracy = %.2f%%\n', ...
            'Elapsed = %s\n', ...
            'Loss data: %s\n', ...
            'Loss chart: %s\n', ...
            'Accuracy data: %s\n', ...
            'Accuracy chart: %s\n', ...
            'Confusion counts: %s\n', ...
            'Confusion percent: %s\n', ...
            'Confusion chart: %s\n', ...
            'Log: %s'], summary.gpu_name, summary.network, preAccuracyText, ...
            finalAccuracy, formatDuration(elapsed), lossCsv, lossPng, accuracyCsv, accuracyPng, ...
            confCountCsv, confPercentCsv, confPng, state.runLogPath));
        if isfile(lossPng)
            winopen(lossPng);
        end
        if isfile(accuracyPng)
            winopen(accuracyPng);
        end
        if isfile(confPng)
            winopen(confPng);
        end
    end

    function showRcResults(elapsed)
        summaryPath = char(fullfile(state.rcOutputDir, 'summary.json'));
        lossCsv = char(fullfile(state.rcOutputDir, 'loss_history.csv'));
        lossPng = char(fullfile(state.rcOutputDir, 'loss_curve.png'));
        accuracyCsv = char(fullfile(state.rcOutputDir, 'accuracy_history.csv'));
        accuracyPng = char(fullfile(state.rcOutputDir, 'accuracy_curve.png'));
        confCountCsv = char(fullfile(state.rcOutputDir, 'confusion_matrix_counts.csv'));
        confPercentCsv = char(fullfile(state.rcOutputDir, 'confusion_matrix_percent.csv'));
        confPng = char(fullfile(state.rcOutputDir, 'confusion_matrix.png'));
        deviceResponseCsv = char(fullfile(state.rcOutputDir, 'device_response_used.csv'));
        featurePreviewCsv = char(fullfile(state.rcOutputDir, 'reservoir_feature_preview.csv'));
        featureMatrixCsv = char(fullfile(state.rcOutputDir, 'reservoir_feature_matrix.csv'));
        featurePcaCsv = char(fullfile(state.rcOutputDir, 'reservoir_feature_pca_coordinates.csv'));
        deviceResponsePng = char(fullfile(state.rcOutputDir, 'device_response_map.png'));
        featureHeatmapPng = char(fullfile(state.rcOutputDir, 'reservoir_feature_heatmap.png'));
        featurePcaPng = char(fullfile(state.rcOutputDir, 'reservoir_feature_pca.png'));
        paperSummaryPng = char(fullfile(state.rcOutputDir, 'paper_style_summary.png'));
        paperMultitaskPng = char(fullfile(state.rcOutputDir, 'paper_multitask_summary.png'));
        if ~isfile(summaryPath)
            error("Reservoir summary file was not created: %s", summaryPath);
        end
        summary = jsondecode(fileread(summaryPath));
        if isfield(summary, 'task') && strcmp(summary.task, 'paper-multitask')
            mnistAcc = summary.tasks.MNIST.final_test_accuracy_percent;
            emnistAcc = summary.tasks.EMNIST.final_test_accuracy_percent;
            fmnistAcc = summary.tasks.FMNIST.final_test_accuracy_percent;
            plotRcResults(char(fullfile(state.rcOutputDir, 'fmnist_loss_history.csv')), char(fullfile(state.rcOutputDir, 'fmnist_accuracy_history.csv')));
            set(rcStatusBox, "String", sprintf([ ...
                'Paper multitask reservoir computing complete.\n', ...
                'GPU = %s\n', ...
                'Dataset = %s\n', ...
                'Reservoir feature dim = %d\n', ...
                'Pulse coding = %d-bit, %d device states\n', ...
                'Readout = %s\n', ...
                'MNIST digit accuracy = %.2f%%\n', ...
                'EMNIST L/M/S accuracy = %.2f%%\n', ...
                'Fashion 5-class accuracy = %.2f%%\n', ...
                'Elapsed = %s\n', ...
                'Paper multitask summary: %s\n', ...
                'Device response data: %s\n', ...
                'Feature matrix data: %s\n', ...
                'Feature PCA coordinates: %s\n', ...
                'Device response chart: %s\n', ...
                'Feature heatmap: %s\n', ...
                'Feature PCA: %s\n', ...
                'Summary: %s\n', ...
                'Log: %s'], summary.gpu_name, summary.dataset, summary.reservoir_feature_dim, ...
                summary.pulse_bits, summary.num_device_states, summary.readout, ...
                mnistAcc, emnistAcc, fmnistAcc, formatDuration(elapsed), paperMultitaskPng, ...
                deviceResponseCsv, featureMatrixCsv, featurePcaCsv, deviceResponsePng, featureHeatmapPng, featurePcaPng, summaryPath, state.runLogPath));
        else
            plotRcResults(lossCsv, accuracyCsv);
            set(rcStatusBox, "String", sprintf([ ...
                'Reservoir computing run complete.\n', ...
                'GPU = %s\n', ...
                'Dataset = %s\n', ...
                'Reservoir feature dim = %d\n', ...
                'Readout = %s\n', ...
                'Final accuracy = %.2f%%\n', ...
                'Final cross-entropy = %.6f\n', ...
                'Elapsed = %s\n', ...
                'Loss data: %s\n', ...
                'Loss chart: %s\n', ...
                'Accuracy data: %s\n', ...
                'Accuracy chart: %s\n', ...
                'Confusion counts: %s\n', ...
                'Confusion percent: %s\n', ...
                'Confusion chart: %s\n', ...
                'Device response data: %s\n', ...
                'Feature preview data: %s\n', ...
                'Feature matrix data: %s\n', ...
                'Feature PCA coordinates: %s\n', ...
                'Device response chart: %s\n', ...
                'Feature heatmap: %s\n', ...
                'Feature PCA: %s\n', ...
                'Paper-style summary: %s\n', ...
                'Summary: %s\n', ...
                'Log: %s'], summary.gpu_name, summary.dataset, summary.reservoir_feature_dim, ...
                summary.readout, summary.final_test_accuracy_percent, summary.final_test_loss, ...
                formatDuration(elapsed), lossCsv, lossPng, accuracyCsv, accuracyPng, ...
                confCountCsv, confPercentCsv, confPng, deviceResponseCsv, featurePreviewCsv, ...
                featureMatrixCsv, featurePcaCsv, deviceResponsePng, featureHeatmapPng, featurePcaPng, paperSummaryPng, summaryPath, state.runLogPath));
        end
        if isfile(paperMultitaskPng)
            winopen(paperMultitaskPng);
        elseif isfile(paperSummaryPng)
            winopen(paperSummaryPng);
        end
        if isfile(deviceResponsePng)
            winopen(deviceResponsePng);
        end
        if isfile(lossPng)
            winopen(lossPng);
        end
        if isfile(accuracyPng)
            winopen(accuracyPng);
        end
        if isfile(confPng)
            winopen(confPng);
        end
    end

    function plotRcResults(lossCsv, accuracyCsv)
        lossData = readmatrix(lossCsv);
        accData = readmatrix(accuracyCsv);
        cla(rcAx);
        yyaxis(rcAx, 'left');
        plot(rcAx, lossData(:, 1), lossData(:, 2), 'k-', 'LineWidth', 1.5);
        hold(rcAx, 'on');
        plot(rcAx, lossData(:, 1), lossData(:, 3), 'r-', 'LineWidth', 1.5);
        plot(rcAx, lossData(:, 1), lossData(:, 4), 'b-', 'LineWidth', 1.5);
        ylabel(rcAx, 'Cross-Entropy');
        yyaxis(rcAx, 'right');
        plot(rcAx, accData(:, 1), accData(:, 3), 'Color', [0.10 0.55 0.35], 'LineWidth', 1.8);
        ylabel(rcAx, 'Testing Accuracy %');
        grid(rcAx, 'on');
        xlabel(rcAx, 'Epoch');
        legend(rcAx, {'train loss', 'test loss', 'val loss', 'test acc'}, 'Location', 'best');
        title(rcAx, 'Reservoir Computing Results');
    end

    function setActiveStatus(text)
        if state.runKind == "rc"
            set(rcStatusBox, "String", text);
        else
            set(statusBox, "String", text);
        end
    end

    function setRunProgress(fraction, label)
        fraction = min(max(fraction, 0), 1);
        if state.runKind == "rc"
            set(rcProgressPatch, "XData", [0 fraction fraction 0], "YData", [0 0 1 1]);
            set(rcProgressText, "String", ['Run progress: ', char(label)]);
        else
            set(progressPatch, "XData", [0 fraction fraction 0], "YData", [0 0 1 1]);
            set(progressText, "String", ['Run progress: ', char(label)]);
        end
    end

    function stopRunTimer()
        if isfield(state, "runTimer") && ~isempty(state.runTimer) && isvalid(state.runTimer)
            stop(state.runTimer);
            delete(state.runTimer);
        end
        state.runTimer = [];
    end

    function stopRunProcess(showMessage)
        stopRunTimer();
        if isfield(state, "runProcess") && ~isempty(state.runProcess)
            try
                if ~state.runProcess.HasExited
                    pid = int32(state.runProcess.Id);
                    system(sprintf('taskkill /PID %d /T /F >nul 2>&1', pid));
                end
            catch
            end
        end
        state.runProcess = [];
        if showMessage
            setRunProgress(0, "Stopped.");
            setActiveStatus("Run stopped.");
        end
    end

    function closeGui()
        stopRunProcess(false);
        delete(fig);
    end

    function text = readLogText(logPath)
        text = "";
        if isfile(logPath)
            fid = fopen(logPath, "r");
            if fid >= 0
                cleaner = onCleanup(@() fclose(fid));
                raw = fread(fid, "*char")';
                text = string(raw);
                clear cleaner;
            end
        end
    end

    function [nInput, nHide, imageSize, epochs] = readNetworkShape()
        imageSize = getSelectedImageSize();
        nInput = imageSize * imageSize;
        nHide = round(readNumber(hiddenNeuronsEdit, "Hidden neurons"));
        epochs = round(readNumber(epochsEdit, "Epochs"));
        if nHide < 1 || nHide > 1000
            error("Hidden neurons must be an integer between 1 and 1000.");
        end
        if epochs < 1 || epochs > 10000
            error("Epochs must be an integer between 1 and 10000.");
        end
    end

    function imageSize = getSelectedImageSize()
        imageSize = 28;
    end

    function applyNetworkShape(nInput, nHide, epochs)
        paramFile = char(fullfile(rootDir, 'Param.cpp'));
        text = fileread(paramFile);
        text = regexprep(text, 'nInput\s*=\s*\d+\s*;', sprintf('nInput = %d;', nInput), 'once');
        text = regexprep(text, 'nHide\s*=\s*\d+\s*;', sprintf('nHide = %d;', nHide), 'once');
        text = regexprep(text, 'totalNumEpochs\s*=\s*\d+\s*;', sprintf('totalNumEpochs = %d;', epochs), 'once');
        fid = fopen(paramFile, 'w');
        if fid < 0
            error("Could not update Param.cpp.");
        end
        cleaner = onCleanup(@() fclose(fid));
        fwrite(fid, text);
        clear cleaner;
    end

    function applyCppPostMappingMode()
        paramFile = char(fullfile(rootDir, 'Param.cpp'));
        text = fileread(paramFile);
        text = regexprep(text, ...
            'useHardwareInTrainingFF\s*=\s*(true|false)\s*;', ...
            'useHardwareInTrainingFF = false;', 'once');
        text = regexprep(text, ...
            'useHardwareInTrainingWU\s*=\s*(true|false)\s*;', ...
            'useHardwareInTrainingWU = false;', 'once');
        text = regexprep(text, ...
            'useHardwareInTraining\s*=\s*[^;]+;', ...
            'useHardwareInTraining = useHardwareInTrainingFF || useHardwareInTrainingWU;', 'once');
        text = regexprep(text, ...
            'useHardwareInTestingFF\s*=\s*(true|false)\s*;', ...
            'useHardwareInTestingFF = true;', 'once');
        fid = fopen(paramFile, 'w');
        if fid < 0
            error("Could not update Param.cpp post-mapping mode.");
        end
        cleaner = onCleanup(@() fclose(fid));
        fwrite(fid, text);
        clear cleaner;
    end

    function datasetName = prepareSelectedDataset(imageSize)
        value = get(datasetPopup, "Value");
        if value == 1
            datasetName = sprintf('MNIST digits (%dx%d)', imageSize, imageSize);
            datasetDir = char(fullfile(rootDir, 'Datasets', sprintf('MNIST_%dx%d', imageSize, imageSize)));
            prepareDatasetIfNeeded('mnist', imageSize, datasetDir);
        else
            datasetName = sprintf('Fashion-MNIST fashion (%dx%d)', imageSize, imageSize);
            datasetDir = char(fullfile(rootDir, 'Datasets', sprintf('FashionMNIST_%dx%d', imageSize, imageSize)));
            prepareDatasetIfNeeded('fashion', imageSize, datasetDir);
        end
        activateDataset(datasetDir);
    end

    function datasetKey = getSelectedTorchDatasetKey()
        if get(datasetPopup, "Value") == 1
            datasetKey = 'mnist';
        else
            datasetKey = 'fashion';
        end
    end

    function prepareDatasetIfNeeded(datasetKey, imageSize, datasetDir)
        names = datasetFileNames();
        ready = isfolder(datasetDir);
        for idx = 1:numel(names)
            ready = ready && isfile(char(fullfile(datasetDir, names{idx})));
        end
        if ~ready
            set(statusBox, "String", sprintf('Preparing %s %dx%d dataset. This may take a moment...', datasetKey, imageSize, imageSize));
            drawnow;
            command = sprintf('python prepare_neurosim_dataset.py --dataset %s --image-size %d', datasetKey, imageSize);
            [code, output] = system(command);
            if code ~= 0
                error("Dataset preparation failed: %s", output);
            end
        end
    end

    function activateDataset(datasetDir)
        names = datasetFileNames();
        for idx = 1:numel(names)
            source = char(fullfile(datasetDir, names{idx}));
            target = char(fullfile(rootDir, names{idx}));
            if ~isfile(source)
                error("Dataset file missing: %s", source);
            end
            copyfile(source, target);
        end
    end

    function names = datasetFileNames()
        names = {'patch60000_train.txt', 'label60000_train.txt', ...
            'patch10000_test.txt', 'label10000_test.txt'};
    end

    function [csvPath, pngPath, finalAccuracy] = exportAccuracyResults(logPath)
        [epochs, accuracies] = parseAccuracyHistory(readLogText(logPath));
        if isempty(epochs)
            error("No accuracy records were found in the run log.");
        end
        csvPath = char(fullfile(rootDir, 'accuracy_history.csv'));
        pngPath = char(fullfile(rootDir, 'accuracy_curve.png'));
        fid = fopen(csvPath, 'w');
        if fid < 0
            error("Could not write accuracy CSV: %s", csvPath);
        end
        cleaner = onCleanup(@() fclose(fid));
        fprintf(fid, 'epoch,accuracy_percent\n');
        for idx = 1:numel(epochs)
            fprintf(fid, '%d,%.6f\n', epochs(idx), accuracies(idx));
        end
        clear cleaner;
        finalAccuracy = accuracies(end);

        figOut = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 640 480]);
        axOut = axes(figOut);
        plot(axOut, epochs, accuracies, 'k-', 'LineWidth', 1.8);
        hold(axOut, 'on');
        plot(axOut, epochs, accuracies, 'r.', 'MarkerSize', 10);
        grid(axOut, 'on');
        xlabel(axOut, 'Epochs');
        ylabel(axOut, 'Accuracy %');
        title(axOut, 'Neural Network Accuracy');
        ylim(axOut, [max(0, min(accuracies)-5), min(100, max(accuracies)+5)]);
        exportgraphics(figOut, pngPath, 'Resolution', 200);
        close(figOut);
    end

    function [csvPath, pngPath] = exportLossResults(logPath)
        [epochs, trainingLoss, validationLoss, testingLoss] = parseLossHistory(readLogText(logPath));
        if isempty(epochs)
            error("No cross-entropy records were found in the run log.");
        end
        csvPath = char(fullfile(rootDir, 'loss_history.csv'));
        pngPath = char(fullfile(rootDir, 'loss_curve.png'));
        fid = fopen(csvPath, 'w');
        if fid < 0
            error("Could not write loss CSV: %s", csvPath);
        end
        cleaner = onCleanup(@() fclose(fid));
        fprintf(fid, 'epoch,training_cross_entropy,testing_cross_entropy,validation_cross_entropy\n');
        for idx = 1:numel(epochs)
            fprintf(fid, '%d,%.8f,%.8f,%.8f\n', epochs(idx), trainingLoss(idx), testingLoss(idx), validationLoss(idx));
        end
        clear cleaner;

        figOut = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 640 480]);
        axOut = axes(figOut);
        plot(axOut, epochs, trainingLoss, 'k-', 'LineWidth', 1.8);
        hold(axOut, 'on');
        plot(axOut, epochs, testingLoss, 'r-', 'LineWidth', 1.8);
        plot(axOut, epochs, validationLoss, 'b-', 'LineWidth', 1.8);
        grid(axOut, 'on');
        xlabel(axOut, 'Epoch');
        ylabel(axOut, 'Cross-Entropy');
        legend(axOut, {'training', 'testing', 'validation'}, 'Location', 'northeast');
        title(axOut, 'Cross-Entropy');
        exportgraphics(figOut, pngPath, 'Resolution', 200);
        close(figOut);
    end

    function [countCsv, percentCsv, pngPath, finalAccuracy] = exportConfusionResults(logPath)
        matrix = parseConfusionMatrix(readLogText(logPath));
        if isempty(matrix)
            error("No confusion matrix was found in the run log.");
        end
        countCsv = char(fullfile(rootDir, 'confusion_matrix_counts.csv'));
        percentCsv = char(fullfile(rootDir, 'confusion_matrix_percent.csv'));
        pngPath = char(fullfile(rootDir, 'confusion_matrix.png'));
        labels = getClassLabels();
        rowSums = sum(matrix, 2);
        percent = zeros(size(matrix));
        for rowIdx = 1:size(matrix, 1)
            if rowSums(rowIdx) > 0
                percent(rowIdx, :) = matrix(rowIdx, :) / rowSums(rowIdx) * 100;
            end
        end
        finalAccuracy = sum(diag(matrix)) / max(sum(matrix(:)), 1) * 100;
        writeMatrixCsv(countCsv, matrix, labels, 'count');
        writeMatrixCsv(percentCsv, percent, labels, 'percent');

        figOut = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 720 620]);
        axOut = axes(figOut);
        imagesc(axOut, percent);
        axis(axOut, 'image');
        blueMap = [linspace(0.96, 0.05, 256)', linspace(0.98, 0.22, 256)', linspace(1.00, 0.48, 256)'];
        colormap(axOut, blueMap);
        colorbar(axOut);
        title(axOut, sprintf('Accuracy: %.2f%%', finalAccuracy));
        xlabel(axOut, 'Predicted');
        ylabel(axOut, 'True');
        set(axOut, 'XTick', 1:numel(labels), 'XTickLabel', labels, ...
            'YTick', 1:numel(labels), 'YTickLabel', labels, ...
            'TickLength', [0 0]);
        xtickangle(axOut, 45);
        for r = 1:size(percent, 1)
            for c = 1:size(percent, 2)
                if percent(r, c) >= 50
                    color = 'w';
                else
                    color = 'k';
                end
                text(axOut, c, r, sprintf('%.1f%%', percent(r, c)), ...
                    'HorizontalAlignment', 'center', 'Color', color, ...
                    'FontWeight', 'bold', 'FontSize', 8);
            end
        end
        exportgraphics(figOut, pngPath, 'Resolution', 200);
        close(figOut);
    end

    function writeMatrixCsv(path, matrix, labels, valueName)
        fid = fopen(path, 'w');
        if fid < 0
            error("Could not write matrix CSV: %s", path);
        end
        cleaner = onCleanup(@() fclose(fid));
        fprintf(fid, 'true_label');
        for idx = 1:numel(labels)
            fprintf(fid, ',predicted_%s', labels{idx});
        end
        fprintf(fid, '\n');
        for r = 1:size(matrix, 1)
            fprintf(fid, '%s', labels{r});
            for c = 1:size(matrix, 2)
                if strcmp(valueName, 'count')
                    fprintf(fid, ',%d', matrix(r, c));
                else
                    fprintf(fid, ',%.6f', matrix(r, c));
                end
            end
            fprintf(fid, '\n');
        end
        clear cleaner;
    end

    function plotAccuracyResults(csvPath)
        data = readmatrix(csvPath);
        if isempty(data)
            return;
        end
        if size(data, 2) >= 2
            epochs = data(:, 1);
            accuracies = data(:, 2);
        else
            return;
        end
        cla(ax);
        plot(ax, epochs, accuracies, 'k-', 'LineWidth', 1.8);
        hold(ax, 'on');
        plot(ax, epochs, accuracies, 'r.', 'MarkerSize', 10);
        grid(ax, 'on');
        xlabel(ax, 'Epochs');
        ylabel(ax, 'Accuracy %');
        title(ax, 'Neural Network Accuracy');
    end

    function plotLossResults(csvPath)
        data = readmatrix(csvPath);
        if isempty(data) || size(data, 2) < 4
            return;
        end
        cla(ax);
        plot(ax, data(:, 1), data(:, 2), 'k-', 'LineWidth', 1.8);
        hold(ax, 'on');
        plot(ax, data(:, 1), data(:, 3), 'r-', 'LineWidth', 1.8);
        plot(ax, data(:, 1), data(:, 4), 'b-', 'LineWidth', 1.8);
        grid(ax, 'on');
        xlabel(ax, 'Epoch');
        ylabel(ax, 'Cross-Entropy');
        legend(ax, {'training', 'testing', 'validation'}, 'Location', 'northeast');
        title(ax, 'Cross-Entropy');
    end

    function [epochs, accuracies] = parseAccuracyHistory(logText)
        tokens = regexp(char(logText), 'Accuracy at\s+(\d+)\s+epochs\s+is\s+:\s+([0-9.]+)%', 'tokens');
        epochs = [];
        accuracies = [];
        for idx = 1:numel(tokens)
            epochs(end+1, 1) = str2double(tokens{idx}{1}); %#ok<AGROW>
            accuracies(end+1, 1) = str2double(tokens{idx}{2}); %#ok<AGROW>
        end
    end

    function [epochs, trainingLoss, validationLoss, testingLoss] = parseLossHistory(logText)
        tokens = regexp(char(logText), ...
            'CrossEntropy at\s+(\d+)\s+epochs:\s+training=([0-9.eE+-]+)\s+validation=([0-9.eE+-]+)\s+testing=([0-9.eE+-]+)', ...
            'tokens');
        epochs = [];
        trainingLoss = [];
        validationLoss = [];
        testingLoss = [];
        for idx = 1:numel(tokens)
            epochs(end+1, 1) = str2double(tokens{idx}{1}); %#ok<AGROW>
            trainingLoss(end+1, 1) = str2double(tokens{idx}{2}); %#ok<AGROW>
            validationLoss(end+1, 1) = str2double(tokens{idx}{3}); %#ok<AGROW>
            testingLoss(end+1, 1) = str2double(tokens{idx}{4}); %#ok<AGROW>
        end
    end

    function matrix = parseConfusionMatrix(logText)
        text = char(logText);
        token = regexp(text, 'ConfusionMatrix counts begin\s*(.*?)\s*ConfusionMatrix counts end', 'tokens', 'once');
        matrix = [];
        if isempty(token)
            return;
        end
        lines = regexp(strtrim(token{1}), '\r?\n', 'split');
        rows = [];
        for idx = 1:numel(lines)
            values = sscanf(lines{idx}, '%d')';
            if numel(values) == 10
                rows = [rows; values]; %#ok<AGROW>
            end
        end
        if size(rows, 1) == 10 && size(rows, 2) == 10
            matrix = rows;
        end
    end

    function labels = getClassLabels()
        if get(datasetPopup, "Value") == 2
            labels = {'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', ...
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'AnkleBoot'};
        else
            labels = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
        end
    end

    function epoch = parseLatestEpoch(logText)
        epoch = 0;
        tokens = regexp(char(logText), 'Accuracy at\s+(\d+)\s+epochs|NeuroSim progress:\s+epoch\s+(\d+)/|Epoch\s+(\d+)/\d+|RC Epoch\s+(\d+)/\d+', 'tokens');
        if isempty(tokens)
            return;
        end
        for k = 1:numel(tokens)
            parts = tokens{k};
            for p = 1:numel(parts)
                if ~isempty(parts{p})
                    epoch = max(epoch, str2double(parts{p}));
                end
            end
        end
    end

    function total = readTotalEpochs()
        total = 125;
        paramFile = char(fullfile(rootDir, 'Param.cpp'));
        if isfile(paramFile)
            text = fileread(paramFile);
            token = regexp(text, 'totalNumEpochs\s*=\s*(\d+)', 'tokens', 'once');
            if ~isempty(token)
                total = str2double(token{1});
            end
        end
    end

    function value = readParamValue(name, defaultValue)
        value = defaultValue;
        paramFile = char(fullfile(rootDir, 'Param.cpp'));
        if isfile(paramFile)
            text = fileread(paramFile);
            token = regexp(text, [name, '\s*=\s*(\d+)'], 'tokens', 'once');
            if ~isempty(token)
                value = str2double(token{1});
            end
        end
    end

    function eta = estimateRemaining(elapsed, fraction)
        if fraction <= 0.02
            eta = "estimating...";
            return;
        end
        remaining = elapsed * (1 - fraction) / fraction;
        eta = string(formatDuration(remaining));
    end

    function text = formatDuration(secondsValue)
        secondsValue = max(0, secondsValue);
        hoursValue = floor(secondsValue / 3600);
        minutesValue = floor(mod(secondsValue, 3600) / 60);
        secondsOnly = floor(mod(secondsValue, 60));
        if hoursValue > 0
            text = sprintf('%dh %02dm %02ds', hoursValue, minutesValue, secondsOnly);
        else
            text = sprintf('%dm %02ds', minutesValue, secondsOnly);
        end
    end

    function drawPreview(targetAx, ltpPath, ltdPath)
        ltp = localReadCurve(ltpPath);
        ltd = localReadCurve(ltdPath);
        gMin = min([ltp(:,2); ltd(:,2)]);
        gMax = max([ltp(:,2); ltd(:,2)]);
        ltpX = (ltp(:,1) - min(ltp(:,1))) / (max(ltp(:,1)) - min(ltp(:,1)));
        ltdX = (ltd(:,1) - min(ltd(:,1))) / (max(ltd(:,1)) - min(ltd(:,1)));
        ltpY = min(max((ltp(:,2) - gMin) / (gMax - gMin), 0), 1);
        ltdY = min(max((ltd(:,2) - gMin) / (gMax - gMin), 0), 1);
        ltpFit = localLtpModel(ltpX, state.result.paramALTPNorm);
        ltdFit = localLtdModel(ltdX, state.result.paramALTDNorm);

        cla(targetAx);
        plot(targetAx, ltpX, ltpY, "bo", "LineWidth", 1.2); hold(targetAx, "on");
        plot(targetAx, ltpX, ltpFit, "b-", "LineWidth", 2);
        plot(targetAx, ltdX, ltdY, "ro", "LineWidth", 1.2);
        plot(targetAx, ltdX, ltdFit, "r-", "LineWidth", 2);
        grid(targetAx, "on");
        xlabel(targetAx, "Normalized pulse number");
        ylabel(targetAx, "Normalized conductance");
        legend(targetAx, "LTP data", "LTP fit", "LTD data", "LTD fit", "Location", "best");
        title(targetAx, "Fitted LTP/LTD curves");
    end

    function path = resolveDataPath(path)
        path = char(path);
        path = strtrim(path);
        if isempty(path)
            error("Data file path is empty.");
        end
        isWindowsAbsolute = numel(path) >= 3 && path(2) == ':' && (path(3) == '\' || path(3) == '/');
        isUncPath = startsWith(path, '\\');
        if ~(isWindowsAbsolute || isUncPath)
            path = char(fullfile(rootDir, path));
        end
    end

    function data = localReadCurve(file)
        raw = readmatrix(file);
        data = raw(:, 1:2);
        data = data(all(isfinite(data), 2), :);
        data = sortrows(data, 1);
    end

    function y = localLtpModel(x, a)
        b = 1 / (1 - exp(-1 / a));
        y = b * (1 - exp(-x / a));
        y = min(max(y, 0), 1);
    end

    function y = localLtdModel(x, a)
        stateX = 1 - x;
        b = 1 / (1 - exp(-1 / a));
        y = b * (1 - exp(-stateX / a));
        y = min(max(y, 0), 1);
    end

    function openReport()
        report = char(fullfile(rootDir, 'fit_device_result.txt'));
        if isfile(report)
            winopen(report);
        else
            warndlg("No report found yet. Run a fit first.", "Report not found");
        end
    end

    function openRcResults()
        if isfolder(state.rcOutputDir)
            winopen(state.rcOutputDir);
        else
            warndlg("No reservoir results found yet. Run Reservoir Computing first.", "Results not found");
        end
    end
end

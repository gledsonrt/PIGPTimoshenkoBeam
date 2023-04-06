function GP = PhysicsInfoSensorPlacement(GP, AP, sensorTypeList, debugger)
    % Sensor placement optimisation considering the physics-informed kernel
    % Gledson Tondo

    % Debugger
    if nargin < 6; debugger = false; end
    if debugger; figure(); hold on; box on; grid on; end
    
    % Store sensor locations
    sensorLocs = [];
    
    % GP parameters
    nSensors = length(sensorTypeList);
    GP.optim.hypsOpt = log([1, AP.L, AP.EI, AP.kGA, eps*ones(1, nSensors)]);
    xStar = AP.x;    
    
    % Get correct lines
    defLine = size(GP.deflections, 1) + 1;  
    rotLine = size(GP.rotations, 1) + 1;    
    strLine = size(GP.strains, 1) + 1;      
 
    for i = 1:sum(nSensors)
        % Select appropriate sensor type
        switch sensorTypeList{i}
            case 'w'
                sType = 1; sTypeStr = 'deflections'; GPline = defLine; 
            case 'r'
                sType = 2; sTypeStr = 'rotations'; GPline = rotLine;
            case 'e'
                sType = 3; sTypeStr = 'strains'; GPline = strLine;
            otherwise
                error('Valid sensor types are deflections, rotations or strains.');
        end
        
        % Filter possible positions, remove already selected ones
        if isempty(sensorLocs); possibleLocs = xStar;
        else; possibleLocs = xStar(all(xStar ~= sensorLocs', 2)); end
        
        % Create entropy map
        entMap = zeros(size(possibleLocs));
        
        % Loop over possible locations
        for j = 1:length(possibleLocs)
            
            % Unobserved locations
            [~, covEnt] = Predict(possibleLocs(j), GP, 1, false, false);
            
            % Select appropriate variance
            covEnt = covEnt(sType);
            
            % Entropy
            ent = 0.5*(log(2*pi*exp(1)*covEnt));
            entMap(j) = ent;
        end
        
        % Select maximum entropy
        [~, newLoc] = max(entMap);
        
        % Debugger
        if debugger; plot(possibleLocs, entMap); plot(possibleLocs(newLoc), entMap(newLoc), '.'); end 
        
        % Add sensor where max entropy occurs
        sensorLocs = cat(1, sensorLocs, possibleLocs(newLoc));
        
        % Update GP Model
        try
            GP.(sTypeStr){GPline,1} = cat(1, GP.(sTypeStr){GPline,1}, possibleLocs(newLoc));
            GP.(sTypeStr){GPline,2} = cat(1, GP.(sTypeStr){GPline,2}, 0);      
        catch
            GP.(sTypeStr){GPline,1} = possibleLocs(newLoc);
            GP.(sTypeStr){GPline,2} = 0;
        end
    end
    
    % Clean artifical zero-readings
    if any(strcmp(sensorTypeList, 'w')); GP.deflections{defLine, 1} = sort(GP.deflections{defLine, 1}); GP.deflections{defLine, 2} = []; end
    if any(strcmp(sensorTypeList, 'r')); GP.rotations{rotLine, 1} = sort(GP.rotations{rotLine, 1}); GP.rotations{rotLine, 2} = []; end
    if any(strcmp(sensorTypeList, 'e')); GP.strains{strLine, 1} = sort(GP.strains{strLine, 1}); GP.strains{strLine, 2} = []; end
    
end
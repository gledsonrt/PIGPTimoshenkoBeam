function k = GetKernels(generateKernels)
% Generate physics-informed Timoshenko covariance kernels
% Gledson Tondo

    % Check if load or generate kernels
    S = license('checkout', 'symbolic_toolbox');
    if S && exist('generateKernels', 'var') && generateKernels
        generateKernels = true; 
    else
        generateKernels = false;
    end
    
    % Start timer
    tic
    
    if ~generateKernels 
        % Load kernels from data        
        fprintf('Loading pre-saved kernels... ')
        k = load('TimoshenkoKernels.mat');
        k = k.kernels;        
    else
        % Calculate kernels
        fprintf('Generating kernel functions... ')
        
        % Symbols:
        % w - deflection [m]
        % r - rotations [rad]
        % v - shear forces [N]
        % s - strains at a depth z [-]
        % m - moments [Nm]
        % q - forces [N]

        % Symbolic variables
        syms logsigma logtheta logalpha x1 x2 EI kGA z 
        
        % Start kernel structure
        k = struct;
        
        % Field names
        % a - Bending deflections
        % p - Bending rotations
        % s - Bending strains
        % m - Bending moments
        % v - Shear forces
        % q - External loads
        % w - Timoshenko deflections
        % r - Timoshenko rotations
        % e - Timoshenko strains

        % Bernoulli deflections
        k.kaa = exp(logsigma).*exp(-(1/2)*(((x1-x2)^2)/(exp(logtheta)^2)));
        k.kap =                              diff(k.kaa,x2, 1);
        k.kas =                        - z * diff(k.kaa,x2, 2);
        k.kam =                    exp(EI) * diff(k.kaa,x2, 2);
        k.kav =                    exp(EI) * diff(k.kaa,x2, 3);    
        k.kaq =                    exp(EI) * diff(k.kaa,x2, 4);

        % Neutral axis rotations
        k.kpa =                             diff(k.kaa,x1, 1);
        k.kpp =                         diff(diff(k.kaa,x1, 1),x2, 1);
        k.kps =                   - z * diff(diff(k.kaa,x1, 1),x2, 2);
        k.kpm =               exp(EI) * diff(diff(k.kaa,x1, 1),x2, 2);
        k.kpv =               exp(EI) * diff(diff(k.kaa,x1, 1),x2, 3);
        k.kpq =               exp(EI) * diff(diff(k.kaa,x1, 1),x2, 4);

        % Strains
        k.ksa =                        - z * diff(k.kaa,x1, 2);
        k.ksp =                   diff(- z * diff(k.kaa,x1, 2),x2, 1);
        k.kss =             - z * diff(- z * diff(k.kaa,x1, 2),x2, 2);
        k.ksm =         exp(EI) * diff(- z * diff(k.kaa,x1, 2),x2, 2);
        k.ksv =         exp(EI) * diff(- z * diff(k.kaa,x1, 2),x2, 3);
        k.ksq =         exp(EI) * diff(- z * diff(k.kaa,x1, 2),x2, 4);

        % Moments
        k.kma =                    exp(EI) * diff(k.kaa,x1, 2);
        k.kmp =              diff( exp(EI) * diff(k.kaa,x1, 2),x2, 1);
        k.kms =        - z * diff( exp(EI) * diff(k.kaa,x1, 2),x2, 2);
        k.kmm =    exp(EI) * diff( exp(EI) * diff(k.kaa,x1, 2),x2, 2);
        k.kmv =    exp(EI) * diff( exp(EI) * diff(k.kaa,x1, 2),x2, 3);
        k.kmq =    exp(EI) * diff( exp(EI) * diff(k.kaa,x1, 2),x2, 4);

        % Shears
        k.kva =                    exp(EI) * diff(k.kaa,x1, 3);
        k.kvp =              diff( exp(EI) * diff(k.kaa,x1, 3),x2, 1);
        k.kvs =        - z * diff( exp(EI) * diff(k.kaa,x1, 3),x2, 2);
        k.kvm =    exp(EI) * diff( exp(EI) * diff(k.kaa,x1, 3),x2, 2);
        k.kvv =    exp(EI) * diff( exp(EI) * diff(k.kaa,x1, 3),x2, 3); 
        k.kvq =    exp(EI) * diff( exp(EI) * diff(k.kaa,x1, 3),x2, 4);

        % Loads
        k.kqa =                    exp(EI) * diff(k.kaa,x1, 4);
        k.kqp =              diff( exp(EI) * diff(k.kaa,x1, 4),x2, 1);
        k.kqs =        - z * diff( exp(EI) * diff(k.kaa,x1, 4),x2, 2);
        k.kqm =    exp(EI) * diff( exp(EI) * diff(k.kaa,x1, 4),x2, 2);
        k.kqv =    exp(EI) * diff( exp(EI) * diff(k.kaa,x1, 4),x2, 3); 
        k.kqq =    exp(EI) * diff( exp(EI) * diff(k.kaa,x1, 4),x2, 4);

        % Timoshenko deflections
        k.kwa =                 k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 2);
        k.kwp =            diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 2),x2, 1);
        k.kws =      - z * diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 2),x2, 2);
        k.kwm =  exp(EI) * diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 2),x2, 2);
        k.kwv =  exp(EI) * diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 2),x2, 3);
        k.kwq =  exp(EI) * diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 2),x2, 4);
        k.kaw =                 k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 2);
        k.kpw =            diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 2),x1, 1);
        k.ksw =      - z * diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 2),x1, 2);
        k.kmw =  exp(EI) * diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 2),x1, 2);
        k.kvw =  exp(EI) * diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 2),x1, 3);
        k.kqw =  exp(EI) * diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 2),x1, 4);
        k.kww =  k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 2) - ...
                 (exp(EI)/exp(kGA)) * diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 2),x2, 2);

        % Timoshenko Rotations
        k.kra =                 diff(k.kaa,x1, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 3);
        k.krp =            diff(diff(k.kaa,x1, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 3),x2, 1);
        k.krs =      - z * diff(diff(k.kaa,x1, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 3),x2, 2);
        k.krm =  exp(EI) * diff(diff(k.kaa,x1, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 3),x2, 2);
        k.krv =  exp(EI) * diff(diff(k.kaa,x1, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 3),x2, 3);
        k.krq =  exp(EI) * diff(diff(k.kaa,x1, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 3),x2, 4);
        k.kar =                 diff(k.kaa,x2, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 3);
        k.kpr =            diff(diff(k.kaa,x2, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 3),x1, 1);
        k.ksr =      - z * diff(diff(k.kaa,x2, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 3),x1, 2);
        k.kmr =  exp(EI) * diff(diff(k.kaa,x2, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 3),x1, 2);
        k.kvr =  exp(EI) * diff(diff(k.kaa,x2, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 3),x1, 3);
        k.kqr =  exp(EI) * diff(diff(k.kaa,x2, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 3),x1, 4);
        k.krr =  diff(diff(k.kaa,x1, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 3),x2, 1) - ...
                 (exp(EI)/exp(kGA)) * diff(diff(k.kaa,x1, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 3),x2, 3);

        % Timoshenko Strains
        k.kea =                 - z * (diff(k.kaa,x1, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 4));
        k.kep =            diff(- z * (diff(k.kaa,x1, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 4)),x2, 1);
        k.kes =      - z * diff(- z * (diff(k.kaa,x1, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 4)),x2, 2);
        k.kem =  exp(EI) * diff(- z * (diff(k.kaa,x1, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 4)),x2, 2);
        k.kev =  exp(EI) * diff(- z * (diff(k.kaa,x1, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 4)),x2, 3);
        k.keq =  exp(EI) * diff(- z * (diff(k.kaa,x1, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 4)),x2, 4);
        k.kae =                 - z * (diff(k.kaa,x2, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 4));
        k.kpe =            diff(- z * (diff(k.kaa,x2, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 4)),x1, 1);
        k.kse =      - z * diff(- z * (diff(k.kaa,x2, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 4)),x1, 2);
        k.kme =  exp(EI) * diff(- z * (diff(k.kaa,x2, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 4)),x1, 2);
        k.kve =  exp(EI) * diff(- z * (diff(k.kaa,x2, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 4)),x1, 3);
        k.kqe =  exp(EI) * diff(- z * (diff(k.kaa,x2, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 4)),x1, 4);
        k.kee =  - z * (diff(- z * (diff(k.kaa,x1, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 4)),x2, 2) - ...
                 (exp(EI)/exp(kGA)) * diff(- z * (diff(k.kaa,x1, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x1, 4)),x2, 4));

        % Timoshenko rotations x deflections x strains
        k.kwr = diff(k.kaa,x2, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 3) - ...
                (exp(EI)/exp(kGA)) * diff(diff(k.kaa,x2, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 3),x1, 2);
        k.kwe = - z * (diff(k.kaa,x2, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 4)) - ...
                (exp(EI)/exp(kGA)) * diff(- z * (diff(k.kaa,x2, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 4)),x1, 2);
        k.krw = diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 2),x1, 1) - ...
                (exp(EI)/exp(kGA)) * diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 2),x1, 3);
        k.kre = diff(- z * (diff(k.kaa,x2, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 4)),x1, 1) - ...
                (exp(EI)/exp(kGA)) * diff(- z * (diff(k.kaa,x2, 2) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 4)),x1, 3);
        k.kew = -z * (diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 2),x1, 2) - ...
                (exp(EI)/exp(kGA)) * diff(k.kaa - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 2),x1, 4));
        k.ker = -z * (diff(diff(k.kaa,x2, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 3),x1, 2) - ...
                (exp(EI)/exp(kGA)) * diff(diff(k.kaa,x2, 1) - (exp(EI)/exp(kGA)) * diff(k.kaa,x2, 3),x1, 4));

        % To anonymous functions
        fprintf('Converting to annonymous functions... ')
        fNames = fieldnames(k);
        for i = 1:length(fNames)
            k.(fNames{i}) = matlabFunction(simplify(k.(fNames{i})));
        end           

    end
    
    % End timer
    t = toc;
    fprintf('Done! Elapsed time: %1.1fs.\n', t);
end
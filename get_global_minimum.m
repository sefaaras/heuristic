function global_min = get_global_minimum(experiment_name, func_num)
% GET_GLOBAL_MINIMUM Returns the known global minimum values for CEC problems
%
% Usage:
%   global_min = get_global_minimum('cec2014_30', 1)
%   global_min = get_global_minimum('cec2020rw', 15)
%
% Supported experiment names: cec2014_*, cec2017_*, cec2020_*, cec2020rw, cec2021_*, cec2022_*
%
% Note: For constrained problems (CEC2020RW), global minimum refers to the
% best known feasible solution's objective value from literature.

% Extract competition type from experiment name
if contains(experiment_name, 'cec2014')
    competition = 'CEC2014';
elseif contains(experiment_name, 'cec2017')
    competition = 'CEC2017';
elseif contains(experiment_name, 'cec2020rw')
    competition = 'CEC2020RW';
elseif contains(experiment_name, 'cec2020')
    competition = 'CEC2020';
elseif contains(experiment_name, 'cec2021')
    competition = 'CEC2021';
elseif contains(experiment_name, 'cec2022')
    competition = 'CEC2022';
else
    error('Unknown experiment name: %s', experiment_name);
end

switch upper(competition)
    case 'CEC2014'
        % CEC2014 Single Objective Real-Parameter Numerical Optimization

        if func_num >= 1 && func_num <= 30
            global_min = func_num * 100;
        else
            error('CEC2014 function number must be between 1 and 30');
        end
        
    case 'CEC2017'
        % CEC2017 Single Objective Real-Parameter Numerical Optimization

        if func_num >= 1 && func_num <= 30
           global_min = func_num * 100;
        else
            error('CEC2017 function number must be between 1 and 30');
        end
        
    case 'CEC2020'
        % CEC2020 Single Objective Real-Parameter Numerical Optimization
        % Functions F1-F10, global minimum values according to technical report
        global_mins = [
            % Unimodal Function
            100;    % F1: Shifted and Rotated Bent Cigar Function (CEC 2017 F1)
            
            % Basic Functions
            1100;   % F2: Shifted and Rotated Schwefel's Function (CEC 2014 F11)
            700;    % F3: Shifted and Rotated Lunacek bi-Rastrigin Function (CEC 2017 F7)
            1900;   % F4: Expanded Rosenbrock's plus Griewangk's Function (CEC2017 f19)
            
            % Hybrid Functions
            1700;   % F5: Hybrid Function 1 (N=3) (CEC 2014 F17)
            1600;   % F6: Hybrid Function 2 (N=4) (CEC 2017 F16)
            2100;   % F7: Hybrid Function 3 (N=5) (CEC 2014 F21)
            
            % Composition Functions
            2200;   % F8: Composition Function 1 (N=3) (CEC 2017 F22)
            2400;   % F9: Composition Function 2 (N=4) (CEC 2017 F24)
            2500;   % F10: Composition Function 3 (N=5) (CEC 2017 F25)
        ];
 
        if func_num >= 1 && func_num <= 10
            global_min = global_mins(func_num);
        else
            error('CEC2020 function number must be between 1 and 10');
        end
        
    case 'CEC2020RW'
        % CEC2020 Real-World Constrained Optimization Problems
        % Best known feasible solutions from technical report
        global_mins = [
            % Industrial Chemical Processes (RC01-RC07)
            189.3116296;      % RC01: Heat Exchanger Network Design (case 1)
            7049.036954;      % RC02: Heat Exchanger Network Design (case 2)
            -4529.119739;     % RC03: Optimal Operation of Alkylation Unit
            -0.3882604362;    % RC04: Reactor Network Design (RND)
            -400.0056;        % RC05: Haverly's Pooling Problem
            1.863830408;      % RC06: Blending-Pooling-Separation problem
            1.56704510;       % RC07: Propane, Isobutane, n-Butane Nonsharp Separation
            
            % Process Synthesis and Design Problems (RC08-RC14)
            2.0;              % RC08: Process synthesis problem
            2.557654574;      % RC09: Process synthesis and design problem
            1.076543083;      % RC10: Process flow sheeting problem
            99.23846365;      % RC11: Two-reactor Problem
            2.924830553;      % RC12: Process synthesis problem
            26887.0;          % RC13: Process design Problem
            53638.94272;      % RC14: Multi-product batch plant
            
            % Mechanical Engineering Problems (RC15-RC33)
            2994.424465;      % RC15: Weight Minimization of a Speed Reducer
            0.03221300814;    % RC16: Optimal Design of Industrial refrigeration System
            0.01266523278;    % RC17: Tension/compression spring design (case 1)
            5885.332773;      % RC18: Pressure vessel design
            1.670217726;      % RC19: Welded beam design
            263.8958433;      % RC20: Three-bar truss design problem
            0.2352424579;     % RC21: Multiple disk clutch brake design problem
            0.5257687074;     % RC22: Planetary gear train design optimization problem
            16.06986872;      % RC23: Step-cone pulley problem
            2.528791841;      % RC24: Robot gripper problem
            1616.119765;      % RC25: Hydro-static thrust bearing design problem
            35.59231971;      % RC26: Four-stage gear box problem
            524.4507606;      % RC27: 10-bar truss design
            14614.13571;      % RC28: Rolling element bearing
            2964895.417;      % RC29: Gas Transmission Compressor Design (GTCD)
            2.613884058;      % RC30: Tension/compression spring design (case 2)
            0.0;              % RC31: Gear train design Problem
            -30665.53867;     % RC32: Himmelblau's Function
            2.639346970;      % RC33: Topology Optimization
            
            % Power System Problems (RC34-RC44)
            0.0;              % RC34: Optimal Sizing of Single Phase DG
            0.07996385400;    % RC35: Optimal Sizing of DG for Active Power Loss Minimization
            0.04773352900;    % RC36: Optimal Sizing of DG and Capacitors for Reactive Power Loss Minimization
            0.01859356300;    % RC37: Optimal Power flow (Minimization of Active Power Loss)
            2.713936600;      % RC38: Optimal Power flow (Minimization of Fuel Cost)
            2.751590900;      % RC39: Optimal Power flow (Minimization of Active Power Loss and Fuel Cost)
            0.0;              % RC40: Microgrid Power flow (Islanded case)
            0.0;              % RC41: Microgrid Power flow (Grid-connected case)
            0.07702710200;    % RC42: Optimal Setting of Droop Controller (Active Power Loss)
            0.07983597000;    % RC43: Optimal Setting of Droop Controller (Reactive Power Loss)
            -6273.171500;     % RC44: Wind Farm Layout Problem
            
            % Power Electronic Problems (RC45-RC50)
            0.03073936000;    % RC45: SOPWM for 3-level Inverters
            0.02024033500;    % RC46: SOPWM for 5-level Inverters
            0.01278306800;    % RC47: SOPWM for 7-level Inverters
            0.01678753576;    % RC48: SOPWM for 9-level Inverters
            0.009311874180;   % RC49: SOPWM for 11-level Inverters
            0.01505147000;    % RC50: SOPWM for 13-level Inverters
            
            % Livestock Feed Ration Optimization (RC51-RC57)
            4550.851149;      % RC51: Beef Cattle (case 1)
            3348.982149;      % RC52: Beef Cattle (case 2)
            4997.606929;      % RC53: Beef Cattle (case 3)
            4240.548253;      % RC54: Beef Cattle (case 4)
            6696.414512;      % RC55: Dairy Cattle (case 1)
            14746.58000;      % RC56: Dairy Cattle (case 2)
            3213.291701;      % RC57: Dairy Cattle (case 3)
        ];
        
        if func_num >= 1 && func_num <= 57
            global_min = global_mins(func_num);
        else
            error('CEC2020RW function number must be between 1 and 57');
        end
        
    case 'CEC2021'
        % CEC2021 Single Objective Real-Parameter Numerical Optimization
        % Functions F1-F10, global minimum = func_num * 100
        global_mins = [
            % Unimodal Function
            100;    % F1: Shifted and Rotated Bent Cigar Function (CEC 2017 F1)
            
            % Basic Functions
            1100;   % F2: Shifted and Rotated Schwefel's Function (CEC 2014 F11)
            700;    % F3: Shifted and Rotated Lunacek bi-Rastrigin Function (CEC 2017 F7)
            1900;   % F4: Expanded Rosenbrock's plus Griewangk's Function (CEC2017 f19)
            
            % Hybrid Functions
            1700;   % F5: Hybrid Function 1 (N=3) (CEC 2014 F17)
            1600;   % F6: Hybrid Function 2 (N=4) (CEC 2017 F16)
            2100;   % F7: Hybrid Function 3 (N=5) (CEC 2014 F21)
            
            % Composition Functions
            2200;   % F8: Composition Function 1 (N=3) (CEC 2017 F22)
            2400;   % F9: Composition Function 2 (N=4) (CEC 2017 F24)
            2500;   % F10: Composition Function 3 (N=5) (CEC 2017 F25)
        ];
        if func_num >= 1 && func_num <= 10
            global_min = global_mins(func_num);
        else
            error('CEC2021 function number must be between 1 and 10');
        end
        
    case 'CEC2022'
        % CEC2022 Single Objective Real-Parameter Numerical Optimization
        % Functions F1-F12, global minimum values according to technical report
        global_mins = [
            300;   % F1: Shifted and Rotated Zakharov Function
            400;   % F2: Shifted and Rotated Rosenbrock's Function
            600;   % F3: Shifted and Rotated Expanded Scaffer's F6 Function
            800;   % F4: Shifted and Rotated Non-Continuous Rastrigin's Function
            900;   % F5: Shifted and Rotated Levy Function
            1800;  % F6: Hybrid Function 1 (N=3)
            2000;  % F7: Hybrid Function 2 (N=6)
            2200;  % F8: Hybrid Function 3 (N=5)
            2300;  % F9: Composition Function 1 (N=5)
            2400;  % F10: Composition Function 2 (N=4)
            2600;  % F11: Composition Function 3 (N=5)
            2700;  % F12: Composition Function 4 (N=6)
        ];
        
        if func_num >= 1 && func_num <= 12
            global_min = global_mins(func_num);
        else
            error('CEC2022 function number must be between 1 and 12');
        end
        
    otherwise
        error('Unsupported competition: %s. Supported: CEC2014, CEC2017, CEC2020, CEC2020RW, CEC2021, CEC2022', competition);
end

end

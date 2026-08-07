function experiment_config = experiment_factory(experiment_name)
% Grid definition for one benchmark suite at one dimension: which functions run,
% how many repetitions, the box and the FE budget.

    experiment_config = struct();
    experiment_config.name = experiment_name;
    experiment_config.description = '';
    experiment_config.dimensions = [];
    experiment_config.function_numbers = [];
    experiment_config.runs_per_experiment = 51;
    experiment_config.bounds = [-100, 100];

    switch lower(experiment_name)

        case 'cec2014_10'
            experiment_config.description = 'CEC2014 functions in 10 dimensions';
            experiment_config.dimensions = 10;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
                                                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...
                                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30];
            experiment_config.maxFE = 10000 * experiment_config.dimensions;

        case 'cec2014_30'
            experiment_config.description = 'CEC2014 functions in 30 dimensions';
            experiment_config.dimensions = 30;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
                                                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...
                                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30];
            experiment_config.maxFE = 10000 * experiment_config.dimensions;

        case 'cec2014_50'
            experiment_config.description = 'CEC2014 functions in 50 dimensions';
            experiment_config.dimensions = 50;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
                                                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...
                                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30];
            experiment_config.maxFE = 10000 * experiment_config.dimensions;

        case 'cec2014_100'
            experiment_config.description = 'CEC2014 functions in 100 dimensions';
            experiment_config.dimensions = 100;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
                                                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...
                                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30];
            experiment_config.maxFE = 10000 * experiment_config.dimensions;

        case 'cec2017_10'
            experiment_config.description = 'CEC2017 functions in 10 dimensions';
            experiment_config.dimensions = 10;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
                                                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...
                                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30];
            experiment_config.maxFE = 10000 * experiment_config.dimensions;
            experiment_config.bounds = [-100, 100];

        case 'cec2017_30'
            experiment_config.description = 'CEC2017 functions in 30 dimensions';
            experiment_config.dimensions = 30;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
                                                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...
                                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30];
            experiment_config.maxFE = 10000 * experiment_config.dimensions;
            experiment_config.bounds = [-100, 100];

        case 'cec2017_50'
            experiment_config.description = 'CEC2017 functions in 50 dimensions';
            experiment_config.dimensions = 50;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
                                                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...
                                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30];
            experiment_config.maxFE = 10000 * experiment_config.dimensions;
            experiment_config.bounds = [-100, 100];

        case 'cec2017_100'
            experiment_config.description = 'CEC2017 functions in 100 dimensions';
            experiment_config.dimensions = 100;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
                                                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, ...
                                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30];
            experiment_config.maxFE = 10000 * experiment_config.dimensions;
            experiment_config.bounds = [-100, 100];

        case 'cec2020_5'
            experiment_config.description = 'CEC2020 functions in 5 dimensions';
            experiment_config.dimensions = 5;
            % F7 omitted: at D=5 the suite splits it into five 1-variable blocks
            % and its Elliptic block computes pow(10,0/0) -> NaN (source hf05).
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 8, 9, 10];
            experiment_config.maxFE = 50000;
            experiment_config.bounds = [-100, 100];

        case 'cec2020_10'
            experiment_config.description = 'CEC2020 functions in 10 dimensions';
            experiment_config.dimensions = 10;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            experiment_config.maxFE = 1000000;
            experiment_config.bounds = [-100, 100];

        case 'cec2020_15'
            experiment_config.description = 'CEC2020 functions in 15 dimensions';
            experiment_config.dimensions = 15;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            experiment_config.maxFE = 3000000;
            experiment_config.bounds = [-100, 100];

        case 'cec2020_20'
            experiment_config.description = 'CEC2020 functions in 20 dimensions';
            experiment_config.dimensions = 20;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            experiment_config.maxFE = 10000000;
            experiment_config.bounds = [-100, 100];

        case 'cec2021_10'
            experiment_config.description = 'CEC2021 functions in 10 dimensions';
            experiment_config.dimensions = 10;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            experiment_config.maxFE = 200000;
            experiment_config.bounds = [-100, 100];

        case 'cec2021_20'
            experiment_config.description = 'CEC2021 functions in 20 dimensions';
            experiment_config.dimensions = 20;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            experiment_config.maxFE = 1000000;
            experiment_config.bounds = [-100, 100];

        case 'cec2022_10'
            experiment_config.description = 'CEC2022 functions in 10 dimensions';
            experiment_config.dimensions = 10;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
            experiment_config.maxFE = 200000;
            experiment_config.bounds = [-100, 100];

        case 'cec2022_20'
            experiment_config.description = 'CEC2022 functions in 20 dimensions';
            experiment_config.dimensions = 20;
            experiment_config.function_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
            experiment_config.maxFE = 1000000;
            experiment_config.bounds = [-100, 100];

        case 'cec2020rw'
            % Dimension, box and budget are per problem; main.m reads them from
            % Cal_par and derives maxFe from the dimension.
            experiment_config.description = 'CEC2020 Real-World Constrained Optimization Problems';
            experiment_config.dimensions = 0;
            experiment_config.function_numbers = 1:57;
            experiment_config.maxFE = 0;
            experiment_config.bounds = [0, 0];
            experiment_config.use_cal_par = true;

        otherwise
            error('Unknown experiment name: %s\nAvailable experiments:\n  CEC2014: cec2014_10, cec2014_30, cec2014_50, cec2014_100\n  CEC2017: cec2017_10, cec2017_30, cec2017_50, cec2017_100\n  CEC2020: cec2020_5, cec2020_10, cec2020_15, cec2020_20\n  CEC2020RW: cec2020rw\n  CEC2021: cec2021_10, cec2021_20\n  CEC2022: cec2022_10, cec2022_20', experiment_name);
    end

end

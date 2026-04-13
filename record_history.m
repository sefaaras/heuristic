% ----------------------------------------------------------------------- %
% History Recording Helper Function for Optimization Algorithms
% ----------------------------------------------------------------------- %
% This function provides a unified way to record population and fitness
% history with sampling for all optimization algorithms (SOS, GSA, BSA, DSA)
%
% Features:
%   - Fixed 10,000 element history arrays regardless of maxFE
%   - 1/10000 sampling ratio (approximately)
%   - Memory efficient storage
%   - Consistent interface across all algorithms
%
% Usage:
%   [pop_hist, fit_hist, hist_idx] = record_history(current_fe, population, 
%       fitness, pop_hist, fit_hist, hist_idx, sampling_interval, history_size)
%
% Input Parameters:
%   current_fe       - Current function evaluation number
%   population       - Current population matrix (N x D)
%   fitness          - Current fitness values vector (N x 1)
%   pop_hist         - Population history array (history_size x N x D)
%   fit_hist         - Fitness history array (history_size x N)
%   hist_idx         - Current index in history arrays
%   sampling_interval- Sampling interval (calculated as floor(maxFE/10000))
%   history_size     - Maximum history size (typically 10000)
%
% Output Parameters:
%   pop_hist         - Updated population history array
%   fit_hist         - Updated fitness history array
%   hist_idx         - Updated history index
%
% Algorithm Logic:
%   Records history at sampling intervals OR if history array not full yet
%   This ensures we capture early algorithm behavior and sample later stages
% ----------------------------------------------------------------------- %
function [pop_hist, fit_hist, hist_idx] = record_history(current_fe, population, fitness, pop_hist, fit_hist, hist_idx, sampling_interval, history_size)
    
    % Record history only at sampling intervals or if we haven't filled the history yet
    if mod(current_fe, sampling_interval) == 0 || hist_idx <= history_size
        if hist_idx <= history_size
            pop_hist(hist_idx, :, :) = population;
            fit_hist(hist_idx, :) = fitness;
            hist_idx = hist_idx + 1;
        end
    end
    
end

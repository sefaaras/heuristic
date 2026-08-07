function [fe_to_target, err_at_target, targets] = target_hits(curve, global_min)

    targets       = 10 .^ (8:-0.25:-8);
    fe_to_target  = nan(1, numel(targets));
    err_at_target = nan(1, numel(targets));

    if isempty(curve) || ~isscalar(global_min) || ~isfinite(global_min)
        return;
    end

    err = double(curve(:).') - double(global_min);

    tol = 1e-6 * max(1, abs(double(global_min)));
    bad = find(err < -tol, 1, 'first');
    if ~isempty(bad)
        err = err(1:bad-1);
    end
    if isempty(err)
        return;
    end

    err = cummin(err);

    for k = 1:numel(targets)
        idx = find(err <= targets(k), 1, 'first');
        if isempty(idx)
            break;
        end
        fe_to_target(k)  = idx;
        err_at_target(k) = err(idx);
    end
end

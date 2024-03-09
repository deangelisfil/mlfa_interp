function f = wrapper(mlfa_interp_generator, target)
    % mlfa_interp_generator is an instance of Mlfa_interp_generator with
    % method err_numerical_approximation_from_parameters
    % target is either "f" or "delta_f" determining whether f_
    % is "f_l" or "delta_f_l".
    f = @f_;
    function res = f_(varargin)
        % varargin is x1, ..., xD, l
        % Note that err is the first output to 
        l = py.int(varargin{end});
        varargin(end) = []; % remove 'l' from varargin
        x = py.numpy.array([varargin{:}]);
        if size(varargin{1}, 1) == 1
            % resize to column vector
            x = x.reshape(py.int(1), py.int(-1));
        end
        res = mlfa_interp_generator.data_generator.err_numerical_approximation_from_parameters(...
            l, x, mlfa_interp_generator.numerical_method_L( ...
                    mlfa_interp_generator.Lstart + 1: end)); % + 1 to ensure that indexing between Python and Matlab is correct. 
        if target == "f"
            res = double(res{1}); % y is in first position of res
        elseif target == "delta_f"
            res = double(res{2}); % err is in second position of res
        else
            error('ValueError');
        end
    end
end

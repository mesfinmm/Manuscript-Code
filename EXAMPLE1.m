function Example1_merged()
% =====================================================================
% Example 1: whole-domain L2 and L-infinity errors with convergence orders
%
%   D_t^alpha u - u_xx + int_0^t e^{xt} u(x,s) ds = f(x,t)
%   u(x,0)=0,  u(0,t)=t^alpha,  u(1,t)=e t^alpha
%   Exact: u(x,t) = e^x t^alpha
%   A = 1, C = 0, lambda = 1, K(x,t,s) = e^{xt}
% =====================================================================
    clear; clc; close all;

    alphas = [0.4];                    
    Ms     = [8, 16, 32, 64];          
    L = 1.0;  T = 1.0;

    fprintf('============================================================\n');
    fprintf('  EXAMPLE 1:  alpha = %.2f  |  L_inf and L2 errors\n', alphas(1));
    fprintf('============================================================\n');

    for a = 1:numel(alphas)
        alpha = alphas(a);
        r     = 2 / alpha;             % Grading parameter for mesh

        fprintf('\n  alpha = %.2f,  r = %.4f\n', alpha, r);
        fprintf('   M  |   N  |   L_inf Error   | Order_inf |    L2 Error   | Order_L2\n');
        fprintf('  ----+------+-----------------+-----------+---------------+----------\n');

        E_Linf  = zeros(size(Ms));
        E_L2    = zeros(size(Ms));
        ord_inf = nan(size(Ms));
        ord_L2  = nan(size(Ms));

        for m = 1:numel(Ms)
            M = Ms(m);
            N = M; 

            % Run solver and get both errors
            [E_Linf(m), E_L2(m)] = run_solver(M, N, T, L, alpha, r);

            % Calculate convergence orders
            if m > 1
                ord_inf(m) = log(E_Linf(m-1)/E_Linf(m)) / log(Ms(m)/Ms(m-1));
                ord_L2(m)  = log(E_L2(m-1)/E_L2(m))  / log(Ms(m)/Ms(m-1));
            end

            % Print results
            if m == 1
                fprintf('  %3d | %4d | %.4e      |    --     | %.4e      |    --\n', ...
                        M, N, E_Linf(m), E_L2(m));
            else
                fprintf('  %3d | %4d | %.4e      |  %5.3f    | %.4e      |  %5.3f\n', ...
                        M, N, E_Linf(m), ord_inf(m), E_L2(m), ord_L2(m));
            end
        end
    end
end 

function [err_Linf, err_L2] = run_solver(M, N, T, L, alpha, r)
    h = L / M;
    x = linspace(0, L, M+1)';
    t = T * ((0:N)/N).^r;
    t(N+1) = T;
    tau = diff(t);

    % ---- IC/BC ----
    U = zeros(M+1, N+1);
    U(1,   :) = t.^alpha;              % u(0,t) = t^alpha
    U(M+1, :) = exp(1) * t.^alpha;     % u(1,t) = e t^alpha
    U(:,   1) = 0;                     % u(x,0) = 0

    sigma   = alpha / 2;
    A_const = 1.0;  C_const = 0.0;  lambda = 1.0;

    for n = 1:N
        t_star = (1 - sigma)*t(n+1) + sigma*t(n);

        g    = compute_weights(n, t, t_star, alpha);
        g_nn = g(n);

        tau_star = [tau(1:n-1), (1 - sigma)*tau(n)];
        W        = compute_Simpson_weights(tau_star);
        W_n      = W(end);
        W_hist   = W(1:end-1);

        % Kernel K = e^{xt}
        K_star  = exp(x * t_star);
        K_plus  = K_star(3:M+1);
        K_mid   = K_star(2:M);
        K_minus = K_star(1:M-1);

        C_plus  = (K_plus  - K_mid) / 12;
        C_minus = (K_minus - K_mid) / 12;

        common_base = g_nn + (1 - sigma)*C_const + lambda*(1 - sigma)*W_n*K_mid;

        nu      = M - 1;
        main_d  = zeros(nu, 1);
        lower_d = zeros(nu-1, 1);
        upper_d = zeros(nu-1, 1);
        rhs     = zeros(nu, 1);

        for i = 1:nu
            im = i + 1;

            Aip = (1/12)*common_base(i) - (1 - sigma)*A_const/h^2 + lambda*(1 - sigma)*W_n*C_plus(i);
            Bi  = (10/12)*common_base(i) + 2*(1 - sigma)*A_const/h^2;
            Aim = (1/12)*common_base(i) - (1 - sigma)*A_const/h^2 + lambda*(1 - sigma)*W_n*C_minus(i);

            fp = calc_f(x(im+1), t_star, alpha);
            fc = calc_f(x(im),   t_star, alpha);
            fm = calc_f(x(im-1), t_star, alpha);
            rhs_i = (1/12)*fp + (10/12)*fc + (1/12)*fm;

            rhs_i = rhs_i + g(1) * apply_H(U(im-1:im+1, 1));
            for jj = 2:n
                coef = g(jj) - g(jj-1);
                rhs_i = rhs_i + coef * apply_H(U(im-1:im+1, jj));
            end

            del2_prev = (U(im+1, n) - 2*U(im, n) + U(im-1, n)) / h^2;
            HUn1      = apply_H(U(im-1:im+1, n));
            rhs_i     = rhs_i + sigma*A_const*del2_prev - sigma*C_const*HUn1;

            rhs_i = rhs_i - lambda*sigma*W_n * ...
                ( K_mid(i)*HUn1 + C_minus(i)*U(im-1, n) + C_plus(i)*U(im+1, n) );

            volterra_H = 0;  volterra_Um = 0;  volterra_Up = 0;
            for kk = 1:n
                volterra_H  = volterra_H  + W_hist(kk)*apply_H(U(im-1:im+1, kk));
                volterra_Um = volterra_Um + W_hist(kk)*U(im-1, kk);
                volterra_Up = volterra_Up + W_hist(kk)*U(im+1, kk);
            end
            rhs_i = rhs_i - lambda * ( K_mid(i)*volterra_H ...
                                     + C_minus(i)*volterra_Um ...
                                     + C_plus(i)*volterra_Up );

            if i == 1,  rhs_i = rhs_i - Aim * U(1,   n+1); end
            if i == nu, rhs_i = rhs_i - Aip * U(M+1, n+1); end

            rhs(i)    = rhs_i;
            main_d(i) = Bi;
            if i > 1,  lower_d(i-1) = Aim; end
            if i < nu, upper_d(i)   = Aip; end
        end

        U(2:M, n+1) = tridiag_solve(lower_d, main_d, upper_d, rhs);
    end

    % ---- Exact solution & Error matrix ----
    U_ex = exp(x) * (t.^alpha);
    E    = U - U_ex;                       

    % ---- L-infinity Error ----
    err_Linf = max(abs(E(:)));

    % ---- Whole-domain L2 norm 
    w_x = [h/2; h*ones(M-1, 1); h/2];          
    L2_squared = 0;
    for n = 1:N
        e_n   = E(:, n);
        e_np1 = E(:, n+1);
        tau_n = t(n+1) - t(n);

        integral_n   = sum(w_x .* e_n.^2);
        integral_np1 = sum(w_x .* e_np1.^2);

        L2_squared = L2_squared + (tau_n/2) * (integral_n + integral_np1);
    end
    err_L2 = sqrt(L2_squared);
end


function H_V = apply_H(V)
    H_V = (1/12)*V(1) + (10/12)*V(2) + (1/12)*V(3);
end

function g = compute_weights(n, t, t_star, alpha)
    g     = zeros(n, 1);
    tau_l = diff(t(1:n+1));
    p1 = 1 - alpha;
    p2 = 2 - alpha;

    a = zeros(n, 1);
    for j = 1:n-1
        U1   = t_star - t(j);
        a(j) = -U1^p1 * expm1(p1 * log1p(-tau_l(j)/U1)) / gamma(2-alpha);
    end
    a(n) = (t_star - t(n))^p1 / gamma(2-alpha);

    b = zeros(n-1, 1);
    for j = 1:n-1
        t_mid = 0.5*(t(j) + t(j+1));
        U1    = t_star - t(j);
        tj    = tau_l(j);
        denom = t(j+2) - t(j);
        d1    = -U1^p1 * expm1(p1 * log1p(-tj/U1));
        term1 = (t_star - t_mid) * d1 / p1;
        d2    = -U1^p2 * expm1(p2 * log1p(-tj/U1));
        term2 = d2 / p2;
        b(j)  = 2 / (gamma(1-alpha)*denom) * (term1 - term2);
    end

    if n == 1
        g(1) = a(1) / tau_l(1);
    else
        g(1) = (a(1) - b(1)) / tau_l(1);
        for j = 2:n-1
            g(j) = (a(j) + b(j-1) - b(j)) / tau_l(j);
        end
        g(n) = (a(n) + b(n-1)) / tau_l(n);
    end
end

function W = compute_Simpson_weights(tau_star)
    n = length(tau_star);
    W = zeros(1, n+1);
    if n == 1
        W(1) = tau_star(1)/2;  W(2) = tau_star(1)/2;  return;
    end
    if mod(n, 2) == 0
        W(1)   = (tau_star(1)+tau_star(2))*(2*tau_star(1)-tau_star(2)) / (6*tau_star(1));
        W(n+1) = (tau_star(n-1)+tau_star(n))*(2*tau_star(n)-tau_star(n-1)) / (6*tau_star(n));
        for j = 1:n/2
            k = 2*j - 1;
            W(k+1) = (tau_star(k)+tau_star(k+1))^3 / (6*tau_star(k)*tau_star(k+1));
        end
        for j = 1:n/2 - 1
            k = 2*j;
            W(k+1) = (tau_star(k-1)+tau_star(k))*(2*tau_star(k)-tau_star(k-1)) / (6*tau_star(k)) ...
                   + (tau_star(k+1)+tau_star(k+2))*(2*tau_star(k+1)-tau_star(k+2)) / (6*tau_star(k+1));
        end
    else
        n_S    = n - 1;
        tau_S  = tau_star(1:n_S);
        h_last = tau_star(n);
        W(1)     = (tau_S(1)+tau_S(2))*(2*tau_S(1)-tau_S(2)) / (6*tau_S(1));
        W(n_S+1) = (tau_S(n_S-1)+tau_S(n_S))*(2*tau_S(n_S)-tau_S(n_S-1)) / (6*tau_S(n_S));
        for j = 1:n_S/2
            k = 2*j - 1;
            W(k+1) = (tau_S(k)+tau_S(k+1))^3 / (6*tau_S(k)*tau_S(k+1));
        end
        for j = 1:n_S/2 - 1
            k = 2*j;
            W(k+1) = (tau_S(k-1)+tau_S(k))*(2*tau_S(k)-tau_S(k-1)) / (6*tau_S(k)) ...
                   + (tau_S(k+1)+tau_S(k+2))*(2*tau_S(k+1)-tau_S(k+2)) / (6*tau_S(k+1));
        end
        W(n_S+1) = W(n_S+1) + h_last/2;
        W(n+1)   = h_last/2;
    end
end

function sol = tridiag_solve(a, b, c, d)
    n = length(b);
    cp = zeros(n,1);  dp = zeros(n,1);
    cp(1) = c(1)/b(1);  dp(1) = d(1)/b(1);
    for i = 2:n-1
        m = b(i) - a(i-1)*cp(i-1);
        if abs(m) < eps, m = eps; end
        cp(i) = c(i)/m;
        dp(i) = (d(i) - a(i-1)*dp(i-1))/m;
    end
    m = b(n) - a(n-1)*cp(n-1);
    if abs(m) < eps, m = eps; end
    dp(n) = (d(n) - a(n-1)*dp(n-1))/m;
    sol = zeros(n,1);  sol(n) = dp(n);
    for i = n-1:-1:1
        sol(i) = dp(i) - cp(i)*sol(i+1);
    end
end

function f = calc_f(x, t, alpha)
    f = gamma(1+alpha)*exp(x) - t^alpha*exp(x) ...
        + exp(x*(1+t)) * t^(1+alpha)/(1+alpha);
end
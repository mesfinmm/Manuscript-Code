function Example3final()
% =====================================================================
% Example 3:
%
%   D_t^alpha u - u_xx + int_0^t x*t*u(x,s) ds = f(x,t)
%   u(x,0) = 0,  u(0,t) = 0,  u(1,t) = t^alpha*cos(1)
%   Exact: u(x,t) = t^alpha * sin( (1 + pi/2) * x )
%   A = 1, C = 0, lambda = 1,  K(x,t,s) = x*t   
% =====================================================================
    clear; clc; close all;

    alphas = [0.3];
    Ms     = [50,100,200];
    L = 1.0;  T = 1.0;
    beta = 1 + pi/2;                       

    nA = numel(alphas); nM = numel(Ms);
    Errors = zeros(nA, nM);
    Errors_L2 = zeros(nA, nM);
    Rates = nan(nA, nM);
    Rates_L2 = nan(nA, nM);

    fprintf('============================================\n');
    fprintf('  EXAMPLE 3  (Oscillatory Behaviour)\n');
    fprintf('  K = x*t, A = 1, C = 0, lambda = 1\n');
    fprintf('  u_exact = t^alpha * sin((1+pi/2) x)\n');
    fprintf('============================================\n');

    for a = 1:nA
        alpha = alphas(a);
        r = (2)/(alpha);       

        fprintf('\n--------------------------------------------\n');
        fprintf('  alpha = %.2f,  r = %.4f\n', alpha, r);
        fprintf('--------------------------------------------\n');

        for m = 1:nM
            M = Ms(m);
            N = M;                          

            % ---- Meshes ----
            h = L / M;
            x = linspace(0, L, M+1)';       
            t = T * ((0:N)/N).^r;          
            t(N+1) = T;
            tau = diff(t);

            % ---- Solution array with IC/BC ----
            U = zeros(M+1, N+1);
            U(1,   :) = 0;                                % u(0,t) = 0
            U(M+1, :) = (t.^alpha) * cos(1);              % u(1,t) = t^alpha cos(1) 
            U(:,   1) = 0;                                % u(x,0) = 0

            sigma   = alpha / 2;
            A_const = 1.0;  C_const = 0.0;  lambda = 1.0;

            % ---- Time stepping ----
            for n = 1:N
                t_star = (1 - sigma)*t(n+1) + sigma*t(n);

                % 1. L2-1_sigma weights
                g    = compute_L2_1sigma_weights(n, t, t_star, alpha);
                g_nn = g(n);

                % 2. Graded Simpson weights 
                tau_star = [tau(1:n-1), (1 - sigma)*tau(n)];
                W        = compute_Simpson_weights(tau_star);
                W_n      = W(end);
                W_hist   = W(1:end-1);          

                % 3. Kernel values at t_star.
                %    K(x,t,s) = x*t .
                K_star  = x * t_star;            
                K_plus  = K_star(3:M+1);
                K_mid   = K_star(2:M);
                K_minus = K_star(1:M-1);

                % 4. Commutator coefficients 
                C_plus  = (K_plus  - K_mid) / 12;
                C_minus = (K_minus - K_mid) / 12;

                % 5. Base quantity 
                common_base = g_nn + (1 - sigma)*C_const ...
                            + lambda*(1 - sigma)*W_n*K_mid;

                % 6. Assemble tridiagonal system
                nu      = M - 1;
                main_d  = zeros(nu, 1);
                lower_d = zeros(nu-1, 1);
                upper_d = zeros(nu-1, 1);
                rhs     = zeros(nu, 1);

                for i = 1:nu
                    im = i + 1;

                    % ---- LHS coefficients 
                    Aip = (1/12)*common_base(i) - (1 - sigma)*A_const/h^2 ...
                        + lambda*(1 - sigma)*W_n*C_plus(i);
                    Bi  = (10/12)*common_base(i) + 2*(1 - sigma)*A_const/h^2;
                    Aim = (1/12)*common_base(i) - (1 - sigma)*A_const/h^2 ...
                        + lambda*(1 - sigma)*W_n*C_minus(i);

                    % ---- RHS: H[f^{n-sigma_n}]
                    fp = calc_f(x(im+1), t_star, alpha, beta);
                    fc = calc_f(x(im),   t_star, alpha, beta);
                    fm = calc_f(x(im-1), t_star, alpha, beta);
                    rhs_i = (1/12)*fp + (10/12)*fc + (1/12)*fm;

                    % ---- RHS: L2-1_sigma history 
                    rhs_i = rhs_i + g(1) * apply_H(U(im-1:im+1, 1));
                    for jj = 2:n
                        coef  = g(jj) - g(jj-1);
                        rhs_i = rhs_i + coef * apply_H(U(im-1:im+1, jj));
                    end

                    % ---- RHS: sigma*A*delta_x^2 U^{n-1} - sigma*C*H[U^{n-1}] 
                    del2_prev = (U(im+1, n) - 2*U(im, n) + U(im-1, n)) / h^2;
                    HUn1      = apply_H(U(im-1:im+1, n));
                    rhs_i     = rhs_i + sigma*A_const*del2_prev - sigma*C_const*HUn1;

                    % ---- RHS: -lambda*sigma*W_n * H[K_{i,n} U^{n-1}] 
                    rhs_i = rhs_i - lambda*sigma*W_n * ...
                        ( K_mid(i)*HUn1 + C_minus(i)*U(im-1, n) + C_plus(i)*U(im+1, n) );

                    % ---- RHS: -lambda * sum_k W_hist(k) H[K_{i,k} U^k] ----
                    %    Since K_{i,k} = K_mid(i)
                    volterra_H  = 0;
                    volterra_Um = 0;
                    volterra_Up = 0;
                    for kk = 1:n
                        volterra_H  = volterra_H  + W_hist(kk)*apply_H(U(im-1:im+1, kk));
                        volterra_Um = volterra_Um + W_hist(kk)*U(im-1, kk);
                        volterra_Up = volterra_Up + W_hist(kk)*U(im+1, kk);
                    end
                    rhs_i = rhs_i - lambda * ( K_mid(i)*volterra_H ...
                                             + C_minus(i)*volterra_Um ...
                                             + C_plus(i)*volterra_Up );

                    % ---- Dirichlet BCs ----
                    if i == 1
                        rhs_i = rhs_i - Aim * U(1, n+1);
                    end
                    if i == nu
                        rhs_i = rhs_i - Aip * U(M+1, n+1);
                    end

                    rhs(i)     = rhs_i;
                    main_d(i)  = Bi;
                    if i > 1,  lower_d(i-1) = Aim; end
                    if i < nu, upper_d(i)   = Aip; end
                end

                % ---- Thomas solve ----
                U(2:M, n+1) = tridiag_solve(lower_d, main_d, upper_d, rhs);
            end

            % ---- Error over the whole space-time grid 
            U_ex = sin(beta * x) * (t.^alpha);
            diff_all = abs(U - U_ex);
            
            % L-inf error
            Errors(a, m) = max(diff_all(:));
            
            % L2 norm error 
            L2_squared = 0;
            spatial_weight = [h/2; h*ones(M-1, 1); h/2];
            for n = 1:N
                err_n   = diff_all(:, n);
                err_np1 = diff_all(:, n+1);
                
                tau_n = t(n+1) - t(n);
                
                
                integral_n   = sum(spatial_weight .* err_n.^2);
                integral_np1 = sum(spatial_weight .* err_np1.^2);                
               
                L2_squared = L2_squared + (tau_n/2) * (integral_n + integral_np1);
            end
            
            Errors_L2(a, m) = sqrt(L2_squared);
        end

        for m = 2:nM
            hp = L/Ms(m-1);  hc = L/Ms(m);
            Rates(a, m) = log(Errors(a, m-1)/Errors(a, m)) / log(hp/hc);
            Rates_L2(a, m) = log(Errors_L2(a, m-1)/Errors_L2(a, m)) / log(hp/hc);
        end

        % ---- Print per-alpha table ----
        fprintf('\n   M    |  L-inf Error  |  Order  |  L2 Error     |  Order\n');
        fprintf('  ------+---------------+---------+---------------+--------\n');
        for m = 1:nM
            if isnan(Rates(a, m))
                fprintf('  %4d  |  %.4e   |   --   |  %.4e   |   --\n', ...
                    Ms(m), Errors(a, m), Errors_L2(a, m));
            else
                fprintf('  %4d  |  %.4e   | %5.3f  |  %.4e   | %5.3f\n', ...
                    Ms(m), Errors(a, m), Rates(a, m), Errors_L2(a, m), Rates_L2(a, m));
            end
        end
    end
end 

function H_V = apply_H(V)
    H_V = (1/12)*V(1) + (10/12)*V(2) + (1/12)*V(3);
end

function g = compute_L2_1sigma_weights(n, t, t_star, alpha)

    g     = zeros(n, 1);
    tau_l = diff(t(1:n+1));
    p1 = 1 - alpha;
    p2 = 2 - alpha;

    % a_{n,j}
    a = zeros(n, 1);
    for j = 1:n-1
        U1   = t_star - t(j);
        a(j) = -U1^p1 * expm1(p1 * log1p(-tau_l(j)/U1)) / gamma(2-alpha);
    end
    a(n) = (t_star - t(n))^p1 / gamma(2-alpha);

    % b_{n,j}
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
        W(1) = tau_star(1)/2;
        W(2) = tau_star(1)/2;
        return;
    end

    if mod(n, 2) == 0
        W(1)   = (tau_star(1)+tau_star(2))*(2*tau_star(1)-tau_star(2)) ...
                 / (6*tau_star(1));
        W(n+1) = (tau_star(n-1)+tau_star(n))*(2*tau_star(n)-tau_star(n-1)) ...
                 / (6*tau_star(n));
        for j = 1:n/2
            k = 2*j - 1;
            W(k+1) = (tau_star(k)+tau_star(k+1))^3 / (6*tau_star(k)*tau_star(k+1));
        end
        for j = 1:n/2 - 1
            k = 2*j;
            W(k+1) = (tau_star(k-1)+tau_star(k))*(2*tau_star(k)-tau_star(k-1)) ...
                     / (6*tau_star(k)) ...
                   + (tau_star(k+1)+tau_star(k+2))*(2*tau_star(k+1)-tau_star(k+2)) ...
                     / (6*tau_star(k+1));
        end
    else
        n_S    = n - 1;
        tau_S  = tau_star(1:n_S);
        h_last = tau_star(n);

        W(1)     = (tau_S(1)+tau_S(2))*(2*tau_S(1)-tau_S(2)) / (6*tau_S(1));
        W(n_S+1) = (tau_S(n_S-1)+tau_S(n_S))*(2*tau_S(n_S)-tau_S(n_S-1)) ...
                   / (6*tau_S(n_S));
        for j = 1:n_S/2
            k = 2*j - 1;
            W(k+1) = (tau_S(k)+tau_S(k+1))^3 / (6*tau_S(k)*tau_S(k+1));
        end
        for j = 1:n_S/2 - 1
            k = 2*j;
            W(k+1) = (tau_S(k-1)+tau_S(k))*(2*tau_S(k)-tau_S(k-1)) ...
                     / (6*tau_S(k)) ...
                   + (tau_S(k+1)+tau_S(k+2))*(2*tau_S(k+1)-tau_S(k+2)) ...
                     / (6*tau_S(k+1));
        end

        W(n_S+1) = W(n_S+1) + h_last/2;
        W(n+1)   = h_last/2;
    end
end

function sol = tridiag_solve(a, b, c, d)

    n  = length(b);
    cp = zeros(n, 1);
    dp = zeros(n, 1);
    cp(1) = c(1)/b(1);
    dp(1) = d(1)/b(1);
    for i = 2:n-1
        m     = b(i) - a(i-1)*cp(i-1);
        if abs(m) < eps, m = eps; end
        cp(i) = c(i)/m;
        dp(i) = (d(i) - a(i-1)*dp(i-1))/m;
    end
    m     = b(n) - a(n-1)*cp(n-1);
    if abs(m) < eps, m = eps; end
    dp(n) = (d(n) - a(n-1)*dp(n-1))/m;
    sol    = zeros(n, 1);
    sol(n) = dp(n);
    for i = n-1:-1:1
        sol(i) = dp(i) - cp(i)*sol(i+1);
    end
end

function f = calc_f(x, t, alpha, beta)

    term = gamma(1+alpha) + beta^2 * t^alpha ...
         + x * t^(alpha+2) / (alpha+1);
    f    = term * sin(beta*x);
end
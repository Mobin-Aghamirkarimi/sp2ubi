function a = tr37885_element( v )
% Vertically polarised TR 37.885 element with a vertical pattern only (azimuth-omni): v = [theta_3dB, SLA_V, A_m,
% max gain dBi, tilt deg]; tilt applied as theta'' = theta - (tilt - 90) for every azimuth (same as antenna.py).
a = qd_arrayant( 'omni' );
th = 90 - a.elevation_grid * 180 / pi;                 % zenith angle (deg) of each elevation grid point
thpp = th - ( v(5) - 90 );
av = -min( 12 * ( ( thpp - 90 ) / v(1) ).^2, v(2) );
g = v(4) + max( av, -v(3) );                            % dBi
a.Fa = repmat( sqrt( 10.^( g(:) / 10 ) ), 1, numel( a.azimuth_grid ) );
a.Fb = zeros( size( a.Fa ) );
a.name = 'tr37885-element';
end

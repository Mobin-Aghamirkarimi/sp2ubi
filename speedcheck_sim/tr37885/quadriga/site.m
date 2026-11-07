function g1_site( in_file, out_file, ckpt_dir )
% TR 37.885 generator for one site: one fixed QuaDRiGa environment (one layout, both heads,
% all passes of the site as receivers, so all passes see the same spatially consistent environment), patched
% per-ray output, per-ray TR 37.885 scatterer-mobility term (extra Doppler nu given per (pass, segment, cluster,
% ray) by the Python driver, drawn with scatterer_mobility.py), per-(segment, cluster) power rescaling
% of the term-on sum to the term-off power, then QuaDRiGa's unmodified segment merge, separately for term off
% (C1 basis, identical to unpatched QuaDRiGa) and term on.
% Input (.mat, written by quadriga/driver.py): cfg (char, or cell {1 x Ntx}: scenario per transmitter, e.g. P3 head A in
%   LOS and in NLOSv at the same position), heads [3 x Ntx], pos {1xP}[3xS], seg_start {1xP}[1xNs] (1-based),
%   t_snap {1xP}[1xS] (s), nu {1xP}[Ntx x Ns x Cmax x 20] (Hz, independent per link), seed, density, head_el [1x5] (TR 37.885 Table 6.1.4-4/-5
%   element: theta_3dB, SLA_V, A_m, max gain dBi, tilt deg; antenna.py), veh_el (iso, primary) or [1x5]
%   (Table 6.1.4-8 vehicle type 2 element, sensitivity).
% Output (.mat): for pass p and transmitter h (h = 1 .. Ntx): off_{p}_{h}, on_{p}_{h} [paths x S] complex, dl_{p}_{h} [paths x S]
%   (s, term off), dlon_{p}_{h} (s, term on), kf_{p}_{h} [1 x Ns] (dB, builder K per segment), pf_{p}_{h} [Ns x Cmax] (term-on / term-off power before
%   rescaling), nray_{p}_{h}, ls_{p}_{h} [1 x S] (large-scale gain, linear: sum of ray powers incl. both antenna
%   patterns; the per-beacon normaliser of the interface and the path gain of the TR 37.885 link budget).
% ckpt_dir (optional): per-builder checkpoint folder (quadriga/checkpoint.m) outside the run's TMPDIR:
%   the fields each builder adds are stored after its reduce_builder; a restart rebuilds hb as before, loads the valid
%   builders in builder order and computes the others; the caller deletes it after the site record. Output-identical
%   to the call without it. Omitted or '': no checkpoint (as before).
if nargin < 3, ckpt_dir = ''; end
t0 = tic;                                               % progress lines (operational only)
S = load( in_file );
addpath( genpath( fullfile( S.quadriga_path, 'quadriga_src' ) ) );   % constants quadriga.patched_path
rand( 'seed', S.seed ); randn( 'seed', S.seed ); rand( 'state', S.seed ); randn( 'state', S.seed );
s = qd_simulation_parameters;
s.center_frequency = 5.9e9; s.show_progress_bars = 0; s.use_absolute_delays = 1; s.use_3GPP_baseline = 0;
n_pass = numel( S.pos );
n_tx = size( S.heads, 2 );
cfgs = S.cfg; if ischar( cfgs ), cfgs = repmat( {cfgs}, 1, n_tx ); end
l = qd_layout( s ); l.no_tx = n_tx; l.no_rx = n_pass; l.tx_position = S.heads;
if isfield( S, 'head_el' ) && ~isempty( S.head_el )
    a = tr37885_element( S.head_el );
    for k = 1:n_tx, l.tx_array(1,k) = copy( a ); end
end
for p = 1:n_pass
    t = qd_track;
    pos = S.pos{p};
    t.initial_position = pos(:,1);
    t.positions = pos - pos(:,1);
    t.segment_index = double( S.seg_start{p}(:)' );
    t.scenario = repmat( cfgs(:), 1, numel( t.segment_index ) );
    t.name = sprintf( 'Rx%04d', p );
    l.rx_track(1,p) = t;
end
if isfield( S, 'veh_el' ) && ~isempty( S.veh_el )
    av = tr37885_element( S.veh_el );
    for p = 1:n_pass, l.rx_array(1,p) = copy( av ); end
end
hb = l.init_builder; init_sos( hb ); gen_parameters( hb, 5 );
clear l
% Memory: QuaDRiGa's get_channels on a builder array calls each builder in turn and
% concatenates (qd_builder/get_channels.m l. 66-101, no random draws across builders), and every (rx, tx) link lies in
% one builder (the scenario is constant per transmitter along a track). So each builder is processed and reduced to the
% compact outputs before the next, and the per-ray coefficients are never copied (output-identical to the all-builder
% call, g1/check_g1_site_memory.py).
O = struct();
ck = g1_ckpt( 'open', ckpt_dir, in_file, [mfilename( 'fullpath' ) '.m'] );
printf( 'g1_site: %d builders ready, %.1f s\n', numel( hb ), toc( t0 ) );
for ib = 1:numel( hb )
    if hb(ib).no_rx_positions == 0, continue; end
    [ O, hit ] = g1_ckpt( 'load', ck, ib, O );
    if hit
        printf( 'g1_site: builder %d loaded from the checkpoint, %.1f s\n', ib, toc( t0 ) );
        continue;
    end
    c = get_channels( hb(ib), [], 0, [], 1 );
    if ck.on, before = g1_ckpt( 'guard', arrayfun( @(k) c(1,k).name, 1:numel( c ), 'UniformOutput', false ), O ); end
    O = reduce_builder( c, S, O );
    clear c
    if ck.on, g1_ckpt( 'save', ck, ib, O, before ); end
    printf( 'g1_site: builder %d computed, %.1f s\n', ib, toc( t0 ) );
end
save( '-v7', out_file, '-struct', 'O' );               % the checkpoint folder stays: the caller deletes it once the site
%                                                         record is written, so a kill
%                                                         between this save and that record still resumes from it
printf( 'g1_site: saved, %.1f s\n', toc( t0 ) );
end

function O = reduce_builder( c, S, O )
% term on: rotate rays, rescale per cluster to the term-off power, sum; then QuaDRiGa's merge for term off and on
CON = cell( 1, numel( c ) );
for k = 1:numel(c)
    nm = c(1,k).name;                                   % e.g. Tx0001_Rx0003_seg0002 (QuaDRiGa naming)
    tk = sscanf( regexp( nm, 'Tx\d+', 'match', 'once' ), 'Tx%d' );
    rk = sscanf( regexp( nm, 'Rx\d+', 'match', 'once' ), 'Rx%d' );
    sk = sscanf( regexp( nm, 'seg\d+', 'match', 'once' ), 'seg%d' );
    if isempty( sk ), sk = 1; end
    rp = c(1,k).rx_position;                            % segments overlap (for the merge): locate this segment's
    pp = S.pos{rk};                                     % snapshots in the pass by their positions
    [~, i0] = min( sum( ( pp - rp(:,1) ).^2, 1 ) );
    i1 = i0 + c(1,k).no_snap - 1;
    if i1 > size( pp, 2 ) || max( sum( ( pp(:, i0:i1) - rp ).^2, 1 ) ) > 1e-12
        error( 'g1_site: segment %s does not map onto the pass snapshots', nm );
    end
    tt = S.t_snap{rk}( i0 : i1 );                       % snapshot times (s)
    R = c(1,k).par.ray_coeff;                           % [1 x 1 x rays x snaps]
    rc = c(1,k).par.ray_cluster;
    nu = zeros( 1, numel(rc) );
    nuall = S.nu{rk};
    if max( rc ) > size( nuall, 3 ), error( 'g1_site: %d clusters exceed the nu bound %d', max( rc ), size( nuall, 3 ) ); end
    for cl = 2:max(rc)                                  % cluster 1 = the dedicated LOS path: exempt
        idx = find( rc == cl );
        nu( idx ) = reshape( nuall( tk, sk, cl, 1:numel(idx) ), 1, [] );
    end
    rot = exp( 2j*pi*( nu(:) * tt(:)' ) );              % [rays x snaps]
    Ron = reshape( R, size( R, 3 ), size( R, 4 ) ) .* rot;   % explicit shape (squeeze fails with 1 cluster / ray)
    coff = paths_x_snaps( c(1,k).coeff );   % [clusters x snaps]
    con_k = zeros( size( coff ) ); pf = ones( 1, size(coff,1) );
    for cl = 1:size( coff, 1 )
        sm = sum( Ron( rc == cl, : ), 1 );
        p_off = mean( abs( coff(cl,:) ).^2 ); p_on = mean( abs( sm ).^2 );
        if cl > 1 && p_on > 0
            pf(cl) = p_on / p_off;
            sm = sm * sqrt( p_off / p_on );
        end
        con_k( cl, : ) = sm;
    end
    if ~any( nu(:) )                                    % no term: term on is an exact copy of term off (F1 check)
        con_k = coff;
    end
    CON{k} = reshape( con_k, size( c(1,k).coeff ) );
    key = sprintf( '%d_%d', rk, tk );
    if ~isfield( O, ['kf_' key] )
        O.(['kf_' key]) = []; O.(['pf_' key]) = [];
        O.(['lss_' key]) = zeros( 1, size( pp, 2 ) ); O.(['lsc_' key]) = zeros( 1, size( pp, 2 ) );
    end
    % large-scale gain: sum of the individual ray powers (no coherent combination; path loss, shadowing, cluster
    % powers and both antenna patterns), averaged with equal weights where segments overlap
    O.(['lss_' key])(i0:i1) += sum( abs( reshape( R, size(R,3), size(R,4) ) ).^2, 1 );
    O.(['lsc_' key])(i0:i1) += 1;
    O.(['kf_' key])(sk) = c(1,k).par.kf_parset;      % = par.kf after the replacement below (as before)
    O.(['pf_' key])(sk, 1:numel(pf)) = pf;
    O.(['nray_' key]) = numel( rc );
    c(1,k).par = struct( 'kf', c(1,k).par.kf_parset );  % frees the per-ray coefficients of this segment
end
con = copy( c );                                        % now without per-ray data
for k = 1:numel( c )
    con(1,k).coeff = CON{k};
end
clear CON
moff = merge( c, 0.5, 0 ); mon = merge( con, 0.5, 0 );
for k = 1:numel( moff )
    nm = moff(1,k).name;
    tk = sscanf( regexp( nm, 'Tx\d+', 'match', 'once' ), 'Tx%d' );
    rk = sscanf( regexp( nm, 'Rx\d+', 'match', 'once' ), 'Rx%d' );
    key = sprintf( '%d_%d', rk, tk );
    O.(['off_' key]) = paths_x_snaps( moff(1,k).coeff );
    O.(['on_' key])  = paths_x_snaps( mon(1,k).coeff );
    O.(['dl_' key])  = paths_x_snaps( moff(1,k).delay );
    % term on has its own merged delays: QuaDRiGa's merge reorders taps by overlap powers, which the per-ray term
    % changes; never pair term-on coefficients with the term-off delays
    if ~isequal( size( mon(1,k).coeff ), size( mon(1,k).delay ) ), error( 'g1_site: term-on coeff/delay size mismatch' ); end
    O.(['dlon_' key]) = paths_x_snaps( mon(1,k).delay );
    if any( O.(['lsc_' key]) == 0 ), error( 'g1_site: snapshot without segment in %s', nm ); end
    O.(['ls_' key]) = O.(['lss_' key]) ./ O.(['lsc_' key]);
    O = rmfield( O, {['lss_' key], ['lsc_' key]} );
end
end

function y = paths_x_snaps( x )
% [paths x snapshots] from QuaDRiGa's [rx x tx x paths x snaps] or [paths x snaps] arrays (squeeze fails with one path)
if ndims( x ) > 2 || ( size( x, 1 ) == 1 && size( x, 2 ) == 1 )
    y = reshape( x, size( x, 3 ), size( x, 4 ) );
else
    y = x;
end
end

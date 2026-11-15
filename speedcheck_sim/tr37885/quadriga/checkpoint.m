function varargout = checkpoint( op, varargin )
% Per-builder checkpoint of quadriga/site.m. Stores, after each builder's reduce_builder, the
% output fields that builder added; a restarted call rebuilds the builders (every random draw of quadriga/site.m is
% made before its builder loop; get_channels draws none) and loads the valid checkpoints in builder order. The output
% is identical to the call without checkpoints.
%   ck = checkpoint( 'open', dir, in_file, code_file )   dir '' = off. meta.txt = SHA-256 of the input payload (the .mat after
%        its 128-byte header, which carries scipy's creation time), of code_file (quadriga/site.m) and of this file, and the
%        Octave version; a folder whose meta differs (stale, unreadable) is deleted and started anew
%   [O, hit] = checkpoint( 'load', ck, ib, O )            appends builder ib's fields if its file matches its SHA-256 sidecar
%   before = checkpoint( 'guard', names, O )              names: the builder's channel names; refuses a link (rx, tx) that an
%        earlier builder already wrote (then the fields a builder adds would not be all it changed)
%   checkpoint( 'save', ck, ib, O, before )               the fields added since 'guard', in order: temp file, rename, sidecar
% The caller deletes the folder after the site record is written, not quadriga/site.m after
% its save: a kill between that save and the record (the work folder with out.mat is emptied at a restart) still resumes.
% Test hook: SPEEDCHECK_KILL_AFTER_BUILDER = ib sends SIGKILL to this Octave right after builder ib's checkpoint
% (resume test; never set in a primary run).
switch op
    case 'open'
        varargout{1} = ck_open( varargin{:} );
    case 'load'
        [varargout{1}, varargout{2}] = ck_load( varargin{:} );
    case 'guard'
        varargout{1} = ck_guard( varargin{:} );
    case 'save'
        ck_save( varargin{:} );
    otherwise
        error( 'checkpoint: unknown operation %s', op );
end
end

function ck = ck_open( d, in_file, code_file )
ck = struct( 'on', ~isempty( d ), 'dir', d );
if ~ck.on, return; end
want = sprintf( 'site checkpoint v1\nin_payload_sha256 %s\nsite_sha256 %s\ncheckpoint_sha256 %s\noctave %s\n', ...
                sha256_of( in_file, 128 ), sha256_of( code_file, 0 ), sha256_of( [mfilename( 'fullpath' ) '.m'], 0 ), ...
                OCTAVE_VERSION );
mf = fullfile( d, 'meta.txt' );
if exist( d, 'dir' )
    have = '';
    try, have = fileread( mf ); catch, end
    if ~strcmp( have, want )
        printf( 'site: checkpoint meta missing or different: %s deleted\n', d );
        confirm_recursive_rmdir( false, 'local' );
        rmdir( d, 's' );
    end
end
if ~exist( d, 'dir' )
    [ok, msg] = mkdir( d );
    if ~ok, error( 'checkpoint: cannot create %s: %s', d, msg ); end
    write_atomic( mf, want );
end
end

function [ O, hit ] = ck_load( ck, ib, O )
hit = false;
if ~ck.on, return; end
f = fullfile( ck.dir, sprintf( 'b%03d.mat', ib ) ); fs = [f '.sha256'];
if ~exist( f, 'file' ) || ~exist( fs, 'file' ), return; end
if ~strcmp( strtrim( fileread( fs ) ), sha256_of( f, 0 ) )
    printf( 'site: checkpoint of builder %d does not match its SHA-256: recomputed\n', ib );
    return;
end
B = load( f );
nm = B.ckpt_fields__;
for k = 1:numel( nm )
    if isfield( O, nm{k} ), error( 'checkpoint: field %s of builder %d already present', nm{k}, ib ); end
    O.(nm{k}) = B.(nm{k});
end
hit = true;
end

function before = ck_guard( names, O )
before = fieldnames( O );
for k = 1:numel( names )
    key = sprintf( '%d_%d', sscanf( regexp( names{k}, 'Rx\d+', 'match', 'once' ), 'Rx%d' ), ...
                   sscanf( regexp( names{k}, 'Tx\d+', 'match', 'once' ), 'Tx%d' ) );
    if isfield( O, ['kf_' key] ) || isfield( O, ['off_' key] )
        error( 'checkpoint: link %s spans two builders; per-builder checkpoints are not valid', key );
    end
end
end

function ck_save( ck, ib, O, before )
if ~ck.on, return; end
fa = fieldnames( O );
if numel( fa ) < numel( before ) || ~isequal( fa(1:numel( before )), before(:) )
    error( 'checkpoint: builder %d changed the fields of an earlier builder', ib );
end
nm = fa(numel( before ) + 1:end);
B = struct();
for k = 1:numel( nm )
    B.(nm{k}) = O.(nm{k});
end
B.ckpt_fields__ = nm;                                   % the order of the fields (save sorts them by name)
f = fullfile( ck.dir, sprintf( 'b%03d.mat', ib ) );
save( '-binary', [f '.tmp'], '-struct', 'B' );
clear B
move_atomic( [f '.tmp'], f );
write_atomic( [f '.sha256'], sprintf( '%s\n', sha256_of( f, 0 ) ) );
kb = getenv( 'SPEEDCHECK_KILL_AFTER_BUILDER' );
if ~isempty( kb ) && str2double( kb ) == ib
    kill( getpid(), 9 );
end
end

function h = sha256_of( f, skip )
% coreutils, so that no copy of a large file enters Octave's memory
[st, out] = system( sprintf( 'tail -c +%d -- ''%s'' | sha256sum', skip + 1, f ) );
h = strtok( out );
if st ~= 0 || numel( h ) ~= 64 || ~exist( f, 'file' ), error( 'checkpoint: SHA-256 of %s failed', f ); end
end

function write_atomic( f, txt )
fid = fopen( [f '.tmp'], 'w' );
if fid < 0, error( 'checkpoint: cannot write %s', f ); end
fputs( fid, txt ); fclose( fid );
move_atomic( [f '.tmp'], f );
end

function move_atomic( a, b )
[err, msg] = rename( a, b );
if err, error( 'checkpoint: rename %s -> %s failed: %s', a, b, msg ); end
end

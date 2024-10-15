# Using parallel transport unfolding, a manifold estimation technique that share similarities with Isomap,
# to compute path length in essentially the tangential space.
# the crucial test is whether we can detect a difference in the dependence between path length reaction time
# in the actual transition period (where the correlation should be significant), and in a control period
# while the neural state is still 'struck' in the motor planning attractor state.
module ManifoldTrajectories
using ParallelTransportUnfolding

function get_flattened_data(X::Array{T,3}, bins::AbstractVector{Float64}, rt::AbstractVector{Float64};t0=0.0) where T <: Real
    # TODO: Allow smoothing
    nbins,ncells,ntrials = size(X)
    eeidx = fill(0, ntrials)
    idx0 = searchsortedfirst(bins, t0)
    for i in 1:ntrials
        eeidx[i] = searchsortedfirst(bins,rt[i])
    end
    get_flattened_data(X, eeidx;idx0=idx0)
end 

function get_flattened_data(X::Array{T,3}, eeidx::AbstractVector{Int64};idx0=1) where T <: Real
    qidx = fill(0, 2, length(eeidx))
    for i in axes(qidx,2)
        qidx[1,i] = idx0
        qidx[2,i] = eeidx[i]
    end
    get_flattened_data(X, qidx)
end


function get_flattened_data(X::Array{T,3}, qidx::Matrix{Int64}) where T <: Real
    offset = 0
    nbins,ncells,ntrials = size(X)
    Xf = zeros(T, ncells,nbins*ntrials)
    nn = fill(0, ntrials)
    size(qidx,2) == ntrials || error("There are $(size(qidx,2)) trials in `rt`, but $(ntrials) trials in X")
    for i in 1:ntrials
        x = X[qidx[1,i]:qidx[2,i],:,i]
        _nn = size(x,1)
        nn[i] = _nn
        Xf[:,offset+1:offset+_nn] .= permutedims(x)
        offset += _nn
    end
    Xf = Xf[:,1:offset]
    Xf, nn
end

function get_geodesic_path_length(X::Array{T,3}, eeidx::AbstractVector{Int64};idx0=1,kvs...) where T <: Real
    Xf,nn = get_flattened_data(X,eeidx;idx0=idx0)
    Xb,_ = get_flattened_data(X,eeidx;idx0=idx0-maximum(eeidx))
    pq = fit(PTU, Xf;kvs...)
    pl = [pq.model.X[1+sum(nn[1:i-1]), sum(nn[1:i])] for i in 1:length(nn)]
    pqb =fit(PTU, Xb;kvs...)
    plb = [pqb.model.X[1+sum(nn[1:i-1]), sum(nn[1:i])] for i in 1:length(nn)]
    pl,plb
end
end #module


export random_mask

function random_mask(n, m)
    mask = zeros(Bool, n)
    mask[1:m] .= true
    shuffle!(mask)
end


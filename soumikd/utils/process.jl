function get_exc_vxc_rho(vxc_filename::String, exc_filename::String, rho_filename::String, keyword::String)

    section_found = false
    vxc_list = Array{Float64}[]
    exc_list = Array{Float64}[]
    rho_list = Array{Float64}[]

    combined_matrix = Array{Float64}([])

    open(vxc_filename, "r") do file
        for line in eachline(file)
            line = strip(line)

            try
                vxc = parse.(Float64, split(line))
                push!(vxc_list, [vxc[1], vxc[end]])
            catch e
            end
        end
        
    end

    open(exc_filename, "r") do file2

        for line2 in eachline(file2)
            line2 = strip(line2)

            try
                exc = parse.(Float64, split(line2, ','))
                push!(exc_list, [exc[end]])
            catch e
            end
        end
    end

    open(rho_filename, "r") do file3

        for line3 in eachline(file3)
            line3 = strip(line3)
            
            if section_found
                if isempty(line3)
                    break
                end
                
                try
                    rho = parse.(Float64, split(line3))
                    push!(rho_list, [rho[2]]) 
                catch e
                end
                
            elseif occursin(keyword, line3)
                section_found = true
            end
        end
    end



    for (idx, val) in enumerate(vxc_list)
        push!(combined_matrix, val[1][1])
        push!(combined_matrix, rho_list[idx][1])
        push!(combined_matrix, val[end][1])
        push!(combined_matrix, exc_list[idx + length(exc_list) - length(vxc_list)][1])
    end

    combined_matrix = Array{Float64}(transpose(reshape(combined_matrix, 4, :)))

    return combined_matrix

end

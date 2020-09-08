################################################################################
# Preamble
################################################################################
using LinearAlgebra
using Tensors
using QuantumOptics
using Random
using Rotations
using Statistics
using Plots
using Distributed
using SharedArrays
using ProgressMeter
using DataFrames
using CSV
using Distributions
################################################################################
# Conversion functions
################################################################################
function mTtoGHz(x)
    return 0.02802495*x
end
function get_hfiA(Bhyp_A,Gauss=0)
    cosTheta_randA = rand()*2 - 1
    Theta_randA = acos(cosTheta_randA)
    Phi_randA = rand()*2π
    if Gauss==0
        Bh = Bhyp_A
        HFIWe = 1
    elseif Gauss==1
        n = Normal(Bhyp_A, 0.5*Bhyp_A)
        d = truncated(n, 0, 2*Bhyp_A)
        Bh = rand(d)
        HFIWe = (3/(2π*(Bhyp_A^2)))^(3/2)*exp(-(3*Bh^2)/(2*Bhyp_A^2))
    end
    H_Ax = Bh*sin(Theta_randA)*cos(Phi_randA)
    H_Ay = Bh*sin(Theta_randA)*sin(Phi_randA)
    H_Az = Bh*cos(Theta_randA)
    return [H_Ax, H_Ay, H_Az, HFIWe]
end
function rz(x)
    return [cos(x) sin(x) 0; -sin(x) cos(x) 0; 0 0 1]
end
function ry(x)
    return [cos(x) 0 -sin(x); 0 1 0; sin(x) 0 cos(x)]
end
function getrandEuler(Nang)
    alphas,betas,gammas=[],[],[]
    for i in 1:Nang
        cosTheta_randA = rand()*2 - 1
        Theta_randA = acos(cosTheta_randA)
        Phi_randA = rand()*2π
        Psi_randA = rand()*2π
        append!(alphas,Theta_randA)
        append!(betas,Phi_randA)
        append!(gammas,Psi_randA)
    end
    return alphas, betas, gammas
end
################################################################################
# Prepare operators and superoperators
################################################################################
b = SpinBasis(1//2)
I2 = one(b)
r0 = SparseOperator(b⊗b⊗b,Diagonal([0,0,0.5,0,0,0,0.5,0]))

TransformAB =  [1 0 0 0; 0 1/√(2) 1/√(2) 0; 0 1/√(2) -1/√(2) 0; 0 0 0 1]
T_AB = SparseOperator(b⊗b, TransformAB)
T_ABC = tensor(T_AB,I2)

PSingletOperator =  SparseOperator(b⊗b⊗b,Diagonal([0,0,1,0,0,0,1,0]))
PTripletOperator =  SparseOperator(b⊗b⊗b,Diagonal([1,1,0,1,1,1,0,1]))
I8 = one(b⊗b⊗b)
PreSingletOp = tensor(transpose(I8),PSingletOperator)+tensor(transpose(PSingletOperator),I8)
PreTripletOp = tensor(transpose(I8),PTripletOperator)+tensor(transpose(PTripletOperator),I8)
PreSTDOp = tensor(transpose(PTripletOperator),PSingletOperator)+tensor(transpose(PSingletOperator),PTripletOperator)

function prepsuper(kvect)
    kS, kT, kSTD = kvect
    HaberkornSingletRec = (1//2)*kS*PreSingletOp
    HaberkornTripletRec = (1//2)*kT*PreTripletOp
    R_STDSuper = kSTD*PreSTDOp
    return HaberkornSingletRec, HaberkornTripletRec, R_STDSuper
end
################################################################################
# Prepare Hamiltonian
################################################################################
function Hams3_9(consts, bvect, mu_B, gvect, frp3ht, Euler, hfiAvect)

    """Hamiltonian"""

    J_r, J_pp, D, E = consts
    gR, gPC, gP3 = gvect
    Bx, By, Bz = bvect
    H_Ax, H_Ay, H_Az, H_gx, H_gy, H_gz = hfiAvect

    b = SpinBasis(1//2)
    Sx = sigmax(b)/2
    Sy = sigmay(b)/2
    Sz = sigmaz(b)/2
    I2 = identityoperator(b)

    # exchange
    H_ex_r  = -2*J_r*(tensor(I2,Sx,Sx)+tensor(I2,Sy,Sy)+tensor(I2,Sz,Sz))-2*(J_r*frp3ht)*(tensor(Sx,I2,Sx)+tensor(Sy,I2,Sy)+tensor(Sz,I2,Sz))
    H_ex_pp = -2*J_pp*(tensor(Sx,Sx,I2)+tensor(Sy,Sy,I2)+tensor(Sz,Sz,I2))
    H_ex    = H_ex_r + H_ex_pp

    # Zeeman
    H_Zee_PC = gPC*mu_B*(Bx*tensor(I2,Sx,I2)+By*tensor(I2,Sy,I2)+Bz*tensor(I2,Sz,I2))
    H_Zee_P3 = gP3*mu_B*(Bx*tensor(Sx,I2,I2)+By*tensor(Sy,I2,I2)+Bz*tensor(Sz,I2,I2))
    H_Zee_r  = gR *mu_B*(Bx*tensor(I2,I2,Sx)+By*tensor(I2,I2,Sy)+Bz*tensor(I2,I2,Sz))
    H_Zee_pp = H_Zee_PC + H_Zee_P3
    H_Zee    = H_Zee_pp + H_Zee_r

    # zfs
    Dtensormol=Euler*([-D/3+E 0 0; 0 -D/3+E 0; 0 0 2*D/3])*transpose(Euler)
    S2x = tensor(Sx,Sx)
    S2y = tensor(Sy,Sy)
    S2z = tensor(Sz,Sz)
    H_zfs = Dtensormol[1,1]*S2x*S2x+Dtensormol[1,2]*S2x*S2y+Dtensormol[1,3]*S2x*S2z+Dtensormol[2,1]*S2y*S2x+Dtensormol[2,2]*S2y*S2y+Dtensormol[2,3]*S2y*S2z+Dtensormol[3,1]*S2z*S2x+Dtensormol[3,2]*S2z*S2y+Dtensormol[3,3]*S2z*S2z
    H_zfs2 = tensor(H_zfs,I2)

    # hyperfine:
    H_hfi1 = H_Ax*tensor(Sx,I2) + H_Ay*tensor(Sy,I2) + H_Az*tensor(Sz,I2)
    H_hfi_g = H_gx*Sx + H_gy*Sy + H_gz*Sz
    H_hfi = tensor(H_hfi1,I2) + tensor(I2,I2,H_hfi_g)

    # total
    H=H_ex + H_Zee + H_zfs2 + H_hfi
    return H
end
################################################################################
# Run calculation for one combination of constants
################################################################################
function outloop2(parset,freepar,consts,kvect,r0,gvect,f,hypvect,thvect=[0])
    newconsts = mTtoGHz(consts)
    HkS,HkT,HkSTD = prepsuper(kvect)
    r0Liouville =  reshape(Array(r0.data), (size(r0)[1]^2,1))
    dicprep = Dict("parset" => parset, "freepar" => freepar, "kS" => kvect[1], "newconsts" => newconsts, "r0L" => [r0,r0Liouville], "super" => [HkS,HkT,HkSTD], "gvect" => gvect, "f" => f, "hypvect" => hypvect, "thvect" => thvect)
    return dicprep
end
function calcpoint7(Euler,par,dicprep)
    consts=dicprep["newconsts"]
    r0,r0Liouville=dicprep["r0L"]
    gvect=dicprep["gvect"]
    f=dicprep["f"]
    kS=dicprep["kS"]
    HaberkornSingletRec,HaberkornTripletRec,R_STDSuper=dicprep["super"]
    Bhyp_A,Nhyp,Bhyp_g,hyptyp=dicprep["hypvect"]
    thetas=dicprep["thvect"]
    parset=dicprep["parset"]
    freepar=dicprep["freepar"]

    if parset == "Jr"
        consts[1]=par
        B0=freepar
    elseif parset == "B0"
        B0=par
    end
    weights=0
    Solnhyp=[]
    for i in 1:Nhyp
        if hyptyp == "iso_P3"
            hypAvect=[Bhyp_A,Bhyp_A,Bhyp_A,0,0,0]
        elseif hyptyp == "none"
            hypAvect=[0,0,0,0,0,0]
        elseif hyptyp == "iso_P3_iso_gxl"
            hypAvect=[Bhyp_A,Bhyp_A,Bhyp_A,Bhyp_g,Bhyp_g,Bhyp_g]
        elseif hyptyp == "ang_P3_ang_gxl_Gauss"
             hypp3 = get_hfiA(Bhyp_A,1)
             hypgx = get_hfiA(Bhyp_g,1)
             hypAvect= vcat(hypp3[1:3],hypgx[1:3])
             weig = hypp3[4]*hypgx[4]
             weights = weights+weig
        end

        Solnth = []
        for j in 1:length(thetas)
            th = thetas[j]
            Hfield = Hams3_9(consts, [B0*sin(th),0,B0*cos(th)], 13.99624493, gvect, f, Euler, hypAvect)
            HfieldCoupled = dagger(T_ABC)*Hfield*T_ABC
            HfieldCoupledSuper = transpose(I8)⊗HfieldCoupled - transpose(HfieldCoupled)⊗I8
            HandK = +im*HfieldCoupledSuper + HaberkornSingletRec + HaberkornTripletRec + R_STDSuper
            LiouvilleSoln = Array(HandK.data) \ r0Liouville
            Liou2HilbertSoln = reshape(LiouvilleSoln, (size(r0)[1],size(r0)[1]))
            LiouvilleSolnC = real(kS*tr(Array(PSingletOperator.data)*Liou2HilbertSoln))
            end
            append!(Solnth, LiouvilleSolnC)
        end
        append!(Solnhyp, mean(Solnth))
    end
    Solnhyp=Solnhyp*weigths
    return mean(Solnhyp)
end

################################################################################
# Run full computation in parallel
################################################################################
function etraceangs(parlist, dicprep, Nang, Erand, save=0)

    println(save)
    EB = Any[]
    for i in 1:(Nang^3)
        if Erand == "out"
            Er = Array(rand(RotMatrix{3}))
        end
        for j in 1:size(parlist)[1]
            if Erand == "in"
                Er = Array(rand(RotMatrix{3}))
            end
            tup = (Er,parlist[j])
            push!(EB,tup)
        end
    end

    k = size(EB)[1]
    output = SharedArray{Float64, 1}(k)
    p = Progress(k,10);
    update!(p,0)
    jj = Threads.Atomic{Int}(0)
    l = Threads.SpinLock()
    Threads.@threads for i in 1:k
        EBi = EB[i]
        output[i] = calcpoint7(EBi[1], EBi[2], dicprep)
        Threads.atomic_add!(jj, 1)
        Threads.lock(l)
        update!(p, jj[])
        Threads.unlock(l)
    end
    output = reshape(output, (size(parlist)[1],Nang^3))

    m = mean(output,dims=2)
    s = std(output,dims=2)
    df = convert(DataFrame, [parlist m s])
    if save != 0
        CSV.write(save, df, header=false)
        d1 = delete!(df, 1)
        display(plot(d1[!,1],d1[!,2],xaxis=:log))
    end
    return df
end

################################################################################
# Example run
################################################################################
Hgx = mTtoGHz(0.226)
Hp3 = mTtoGHz(1.1)
Bfields = vcat([0], 10 .^(range(-5,stop=0,length=21)))
th1 = acos.(range(1,stop=0,length=10))
jvect = vcat([0], 10 .^(range(-3,stop=2,length=21)))*0.05

a = outloop2("Jr",0,[0,0.05,-1,0],[1.4e-3,1.4e-3,0],r0,[2.00463,1.999,2.002],0,[Hp3,100,Hgx,"ang_P3_ang_gxl_Gauss"],[0])
e = etraceangs(Bfields,a,5,"out","testrun.csv")

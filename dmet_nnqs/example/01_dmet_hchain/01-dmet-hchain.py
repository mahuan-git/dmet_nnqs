#!/usr/bin/env python
'''
DMET_NNQS calculation for 1-dimensional hydrogen chain.
'''
import numpy as np
import scipy.linalg as la

from pyscf import lib
from pyscf.pbc import scf, gto, df, cc

import libdmet.dmet.Hubbard as dmet
from libdmet.system import lattice
from libdmet.basis_transform import make_basis
from libdmet.utils import logger as log
from libdmet.system.fourier import R2k
from libdmet.utils.plot import plot_density_k as plot_density
from dmet_nnqs.nnqs import NNQS
from dmet_nnqs.config import MyConfig

log.verbose = "DEBUG2"
np.set_printoptions(4, linewidth=1000, suppress=False)

## nnqs config settings:
load_model_init = True  # If true, load a pretrained model for first step. If False, start from random model

config_file = "config.yaml"

strong_cfg ={
            'load_model': 1,
            'log_step' : 1,
            'std_dev_tol': 2e-7,
            'optim': {'name': 'AdamW',
                'lr': 1e-5,
                'betas': [0.9, 0.99],
                'eps': '1e-9',
                'weight_decay': 0.0,
                'open_lr_scheduler': False,
                'warmup_step': 100}
            }

weak_cfg ={
            'load_model': 1,
            'log_step' : 5,
            'std_dev_tol': 2e-7,
            'optim': {'name': 'AdamW',
                'lr': 1e-3,
                'betas': [0.9, 0.99],
                'eps': '1e-9',
                'weight_decay': 0.001,
                'open_lr_scheduler': True,
                'warmup_step': 100}
            }


energy_list = []

for dist in [1.0]:
    cell = lattice.HChain(R=dist)
    cell.basis = 'gth-szv'
    cell.pseudo  = 'gth-pade'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')
    nkz = 5
    kmesh = [1, 1, nkz]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts
    
    exxdiv = None
    
    ### ************************************************************
    ### DMET settings 
    ### ************************************************************
    
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = True
    bogoliubov = False
    int_bath = True
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.0
    beta = np.inf
    
    # DMET SCF control
    MaxIter = 20
    u_tol = 1.0e-4
    E_tol = 1.0e-6
    iter_tol = 4
    load_frecord = False

    # DIIS
    adiis = lib.diis.DIIS()
    adiis.space = 4
    diis_start = 3
    dc = dmet.FDiisContext(adiis.space)
    trace_start = 2
    
    # solver and mu fit
    nelec_tol = 5.0e-6
    delta = 0.01
    step = 0.1
   
    # vcor fit
    imp_fit = True
    emb_fit_iter = 500 # embedding fitting
    full_fit_iter = 100
    
    # vcor initialization
    vcor = dmet.vcor_zeros(restricted, bogoliubov, nscsites)
    
    ### ************************************************************
    ### SCF Mean-field calculation
    ### ************************************************************
    
    log.section("\nSolving SCF mean-field problem\n")
    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    gdf.build()
    
    chkfname = 'hchain.chk'
    if restricted ==False:
        kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
    kmf.with_df = gdf
    kmf.with_df._cderi = 'gdf_ints.h5'
    kmf.conv_tol = 1e-12
    kmf.max_cycle = 300
    kmf.chkfile = chkfname
    kmf.kernel()
    print("kmf energy: ", kmf.e_tot)
    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************
    
    log.section("\nPre-process, orbital localization and subspace partition\n")
    kmf = Lat.symmetrize_kmf(kmf)
    dm1 = kmf.make_rdm1()
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao='minao', full_return=True)
    C_ao_iao = Lat.symmetrize_lo(C_ao_iao)
    
    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)
    
    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo)
    
    ### ************************************************************
    ### DMET procedure
    ### ************************************************************
    
    # DMET main loop
    E_old = 0.0
    conv = False
    history = dmet.IterHistory()
    dVcor_per_ele = None
    if load_frecord:
        dmet.SolveImpHam_with_fitting.load("./frecord")
    
    for iter in range(MaxIter):
        log.section("\nDMET Iteration %d\n", iter)
        
        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %20.12f", Mu)
        if restricted == True:
            rho, Mu, res = dmet.RHartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
        else:
            rho, Mu, res = dmet.HartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
        mol_name = f"HChain_{dist:.2f}"
        frag_idx =0

        ## config setting for initial iteration
        config = MyConfig(config_file,strong_cfg = strong_cfg,weak_cfg = weak_cfg)
        ## define checkpoint file to load 
        if iter ==0:
            if load_model_init == True:
                config.load_model = 1
                config.checkpoint_path = "checkpoints/start.pt"
                config.update_dict(config.strong_cfg)
            else:
                config.load_model = 0
        else:
            load_checkpoint_file = "checkpoints/HChain_1.00_0-0-converge.pt"
            config.load_model = 1
            config.checkpoint_path = load_checkpoint_file
            config.update_dict(config.strong_cfg)
        nnqs_solver = NNQS(config = config,mol_name = mol_name,frag_idx = frag_idx ,restricted=restricted,calc_rdm2=False)
        solver = nnqs_solver

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, int_bath=int_bath)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)
    
        log.section("\nsolving impurity problem\n")
        solver_args = {"nelec": min((Lat.ncore+Lat.nval)*2, \
                cell.nelectron*nkpts), \
                "dm0": dmet.foldRho_k(res["rho_k"], basis_k)*2.0}
        rhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args=solver_args, thrnelec=nelec_tol, \
            delta=delta, step=step)
        dmet.SolveImpHam_with_fitting.save("./frecord")
        last_dmu += dmu
        rhoImp, EnergyImp, nelecImp = \
                dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
                lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
                solver=solver, solver_args=solver_args, rebuild_veff=True)
        EnergyImp *= nscsites
        log.result("last_dmu = %20.12f", last_dmu)
        log.result("E(DMET) = %20.12f", EnergyImp)
        solver.twopdm = None
    
        dump_res_iter = np.array([Mu, last_dmu, vcor.param, rhoEmb, basis, rhoImp, \
                C_ao_lo, rho, Lat.getFock(kspace=False)], dtype=object)
        np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter, allow_pickle=True)
        
        log.section("\nfitting correlation potential\n")
        vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, \
                vcor, beta, Filling, MaxIter1=emb_fit_iter,
                MaxIter2=full_fit_iter, method='CG', \
                imp_fit=imp_fit, ytol=1e-7, gtol=1e-4)
    
        if iter >= trace_start:
            # to avoid spiral increase of vcor and mu
            log.result("Keep trace of vcor unchanged")
            vcor_new = dmet.make_vcor_trace_unchanged(vcor_new, vcor)
    
        dVcor_per_ele = la.norm(vcor_new.param - vcor.param) / (len(vcor.param))
        dE = EnergyImp - E_old
        E_old = EnergyImp 
        
        if iter >= diis_start:
            pvcor = adiis.update(vcor_new.param)
            dc.nDim = adiis.get_num_vec()
        else:
            pvcor = vcor_new.param
        
        dVcor_per_ele = la.norm(pvcor - vcor.param) / (len(vcor.param))
        vcor.update(pvcor)
        log.result("trace of vcor: %s", \
                np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2), axis=1))
        
        history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
        history.write_table()
    
        if dVcor_per_ele < u_tol and abs(dE) < E_tol and iter > iter_tol :
            conv = True
            break
    energy_list.append(EnergyImp+kmf.energy_nuc())
    print("Calculated DMET enrgies: ", energy_list)

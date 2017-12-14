import sys
import json
from collections import OrderedDict
import subprocess as sbp
import logging

from tqdm import tqdm

import numpy as np

from pymongo import MongoClient

from pymatgen import Molecule, Structure
from pymatgen.util.testing import PymatgenTest
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.defects import ValenceIonicRadiusEvaluator

from pymatgen.io.lammps.input import LammpsInput
from pymatgen.io.feff.inputs import  Atoms
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.lammps.output import LammpsRun

from matminer.descriptors.structure import StructuralAttribute
from matminer.featurizers.site import OPSiteFingerprint


logger = logging.getLogger("xas_desc")
logger.setLevel(logging.DEBUG)
sh = logging.FileHandler("xas_desc.log")
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)


def get_site_features(s, site_idx):
    
    vire = ValenceIonicRadiusEvaluator(s)
    if np.linalg.norm(s[site_idx].coords - vire.structure[site_idx].coords) > 1e-6:
        raise RuntimeError("Mismatch between input structure and VIRE structure.")
    sa = StructuralAttribute(vire.structure)
    nn = sa.get_neighbors_of_site_with_index(site_idx)
    rn = vire.radii[vire.structure[site_idx].species_string]
    bond_lengths = [s[site_idx].distance_from_point(x.coords) * (vire.radii[x.species_string]/(rn+vire.radii[x.species_string])) for x in nn]
    #bond_lengths = [s[site_idx].distance_from_point(x.coords) for x in nn]
    # Z, valence, coordination number, weighted avg bond length
    return vire.structure[site_idx].specie.number, vire.valences[vire.structure[site_idx].species_string], len(nn),  np.mean(bond_lengths) #, np.std(bond_lengths)


def get_op_site_features(s, site_idx):

    opsf = OPSiteFingerprint()
    f = opsf.featurize(s, site_idx)    
    return f.tolist()


def get_snap_site_features(d):
    feature = []
    data_filename = "lammps_snap.data"
    input_template = "lammps_snap_template.in"
    input_filename = "lammps_snap.in"
    dump_filename =  "dump.sna"
    log_filename = "log.lammps"

    sbp.check_call(["rm", "-f", input_filename])
    sbp.check_call(["rm", "-f", data_filename])    
    sbp.check_call(["rm", "-f", dump_filename])
    sbp.check_call(["rm", "-f", log_filename])

    structure = Structure.from_dict(d["structure"])
    feature.append(structure[d["absorbing_atom"]].specie.number)
    try:
        mol = Molecule.from_dict(d["cluster"])
    except TypeError:
        atoms = Atoms(structure, d["absorbing_atom"], 10.0)
        mol = atoms.cluster
    logger.info(mol.formula)
    lmp_data = LammpsData.from_structure(mol, [[0,25], [0,25],[0,25]], translate=False)    

    lmp_data.write_file(data_filename)
    el_sorted = sorted(mol.composition, key=lambda x:x.atomic_mass)
    cutoff = ""
    weight = ""
    for i, e in enumerate(el_sorted):
        cutoff += " {}".format(float(e.atomic_radius))
        weight += " {}".format(1.0)
        settings = {
            'data_file': data_filename,
            'rcutfac': 1.4, 
            'rfac0': 0.993630,
            'twojmax': 6.0,
            'cutoff': cutoff,
            'weight': weight,
            'dump_file': dump_filename
        }
    lmp_in = LammpsInput.from_file(input_template, settings)
    lmp_in.write_file(input_filename)
    #try:
    logger.info("Running LAMMPS ... ")
    exit_code = sbp.check_call(["./lmp_serial", "-in", input_filename])
    if exit_code != 0:
        logger.error("lammps run failed")
        raise RuntimeError("lammps run failed")                
    logger.info("Processing LAMMPS outputs ... ")
    lmp_run = LammpsRun(data_filename, dump_filename, log_filename)
    t = list(lmp_run.trajectory[0])
    try:
        assert np.linalg.norm(t[2:5]) <= 1e-6
    except AssertionError:
        logger.info("xyz: {}".format(t[2:5]))
        logger.error("assertion failed: first one not at origin")
        raise
    logger.info("# bispectrum coeffs: {}".format(len(t[5:])))
    feature.extend(t[5:])
    return feature


if __name__ == "__main__":

    conn = MongoClient("localhost", port=57003)
    db = conn["feff_km_share"]
    db.authenticate("anon", "anon")
    xas = db["xas"]

    N = 10000 # dump every N

    all_data = {}
    
    i = 0
    cursor = xas.find({}, no_cursor_timeout=True)

    for doc in tqdm(cursor):
        if "xas_id" in doc and doc["spectrum_type"] == "XANES":
            logger.info("{} {} {}".format(doc["xas_id"], doc["spectrum_type"], i))
        
            #xas_structure = Structure.from_dict(doc["structure"])
            try:
                #f = get_site_features(xas_structure, doc["absorbing_atom"])
                f = get_snap_site_features(doc)            
            except:
                logger.error("xas_id {} skipped".format(doc["xas_id"]))
                continue

            all_data[doc["xas_id"]] = {"site": tuple(list(f)),
                                   "spectrum": doc["spectrum"]}
            i = i + 1
            if i%N == 0:
                fname = "snap_data_{}.json".format(i)
                with open(fname, "w") as ff:
                    json.dump(all_data, ff)
                    logger.info("Data dumped to {}".format(fname))

    cursor.close()

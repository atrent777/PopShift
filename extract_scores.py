"""
Extracts docking scores from a collection of docked ligand poses 
created by docking_parallel.py.
    Copyright (C) 2023  Louis G. Smith

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
from enspara import ra
import numpy as np
import multiprocessing as mp
import argparse as ap
import re
from pathlib import Path
import pickle


def extract_score_from_vina_pdbqt(pdbqts, search_re=re.compile(r'^REMARK VINA RESULT')):
    outlist = []
    for pdbqt in pdbqts:
        with pdbqt.open() as f:
            for line in f:
                if search_re.match(line):
                    outlist.append(float(line.split(maxsplit=4)[3]))
                    break
    return np.array(outlist, dtype='f4')


def extract_score_from_smina_pdbqt(pdbqts, search_re=re.compile(r'^REMARK minimizedAffinity')):
    outlist = []
    # print(pdbqts)
    for pdbqt in pdbqts:
        with pdbqt.open() as f:
            for line in f:
                if search_re.match(line):
                    outlist.append(float(line.split(maxsplit=4)[2]))
                    break
    return np.array(outlist, dtype='f4')

def extract_score_from_gnina_sdf(sdfs, search_re=re.compile(r'> <CNNaffinity>'), save_Kd=False):
    outlist, kdList = [], []

    #relevent constants
    R = 1.98720425864083  # cal*k^-1mol^-1, wikipedia table
    T = 310.0  # Kelvin
    unit_scale = 0.001  # by default convert free energy to kilo-energy units.
    rt = R*T*unit_scale
    #print(f'rt: {rt}')

    # print(pdbqts)
    for sdf in sdfs:
        with sdf.open() as f:
            nextLine = False
            for line in f:
                if search_re.match(line):
                    nextLine=True
                elif nextLine:
                    Kd = np.power(10,(-1*float(line)))
                    #print(f'Kd: {Kd}')
                    fe = rt * np.log(Kd)
                    print(f'fe: {fe}\n')
                    outlist.append(fe)
                    kdList.append(Kd)
                    nextLine = False
                    break
    kdListOut = np.array(kdList, dtype='f4')
    #print(f'kdList:\n{kdListOut}')
    return np.array(outlist, dtype='f4'), np.array(kdList, dtype='f4')

def extract_score_from_gnina_sdf_rescore(sdfs, search_re=re.compile(r'> <minimizedAffinity>')):
    outlist = []
    # print(pdbqts)
    for sdf in sdfs:
        with sdf.open() as f:
            nextLine = False
            for line in f:
                if search_re.match(line):
                    nextLine=True
                elif nextLine:
                    outlist.append(float(line))
                    nextLine = False
                    break
    return np.array(outlist, dtype='f4')

number_pattern = re.compile('\d+')


def rip_last_number(p: Path, numpattern=number_pattern):
    return int(numpattern.findall(p.stem)[-1])


def rip_all_numbers(p: Path, numpattern=number_pattern):
    return tuple(map(int, numpattern.findall(p.stem)))


# note, only finds first match, then uses last match of digit pattern.
def rip_number_bytag(p: Path, tag: re.Pattern, numpattern=number_pattern):
    match = tag.search(p.stem)[0]
    return int(numpattern.findall(match)[0])


extract_types = {
    'vina': extract_score_from_vina_pdbqt,
    'smina': extract_score_from_smina_pdbqt,
    'gnina': extract_score_from_gnina_sdf,
    'gnina_rescore': extract_score_from_gnina_sdf_rescore,
}

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('docking_runs', nargs='+',
                        help='Path(s) to docked ligand directories, containing PDBQTs or SDFs.')
    parser.add_argument('--centers', action=ap.BooleanOptionalAction, default=False,
                        help='If thrown, expect that all ligand files for a given ligand will be '
                        'together in one large directory, with state numbers in file '
                        'names (4x mydockrun/myligand/state000123.pdbqt).')
    parser.add_argument('--centers-tag', type=str, default=None,
                        help='If provided, search for provided regex in result file names, '
                        'then turn all numbers from within the match into a state index.'
                        ' Implies "--centers".')
    parser.add_argument('--replicas', action=ap.BooleanOptionalAction, default=False,
                        help='If thrown, expect that there will be an additional level of depth '
                        'of the directory structure named something like "replica0".')
    parser.add_argument('--nprocs', '-n', type=int, default=1,
                        help='number of processors to use for multiprocessing module.')
    parser.add_argument('--result-type', '-t', choices=extract_types.keys(), default='vina',
                        help='Which type of output file you are trying to extract a score from.')
    parser.add_argument('--extract-to', type=Path, default=None,
                        help='If provided, write extracted scores to this directory instead of customary one.')
    parser.add_argument('--rescoreGnina', action=ap.BooleanOptionalAction, default=False,
                        help='If thrown, will use SMINA scores generated by GNINA docking. ')
    parser.add_argument('--save_Kd', action=ap.BooleanOptionalAction, default=False,
                        help='If thrown, will additionally save Kd values generated by GNINA docking. ')
    args = parser.parse_args()
    pool = mp.Pool(args.nprocs)
    if args.rescoreGnina : args.result_type = 'gnina_rescore'
    extractor = extract_types[args.result_type]
    if args.centers_tag:
        tag_pattern = re.compile(args.centers_tag)
    elif args.centers:
        tag_pattern = re.compile('state\d+')
    else:
        tag_pattern = None

    if tag_pattern:
        def result_sorter(x): return rip_number_bytag(x, tag_pattern)
    else:
        result_sorter = rip_all_numbers

    for dock_run in args.docking_runs:
        dock_path = Path(dock_run)
        if args.extract_to:
            out_path = args.extract_to
        else:
            out_path = dock_path.parent / 'extracted_scores' / dock_path.name
        out_path.mkdir(exist_ok=True, parents=True)
        for ligand_path in dock_path.iterdir():
            if args.replicas:
                rep_paths = sorted((rep_path for rep_path in ligand_path.iterdir()),
                                   key=rip_last_number)
                for rep_path in rep_paths:
                    rep_ix = rip_last_number(rep_path)
                    if args.centers or args.centers_tag:
                        result_paths = sorted((sample_path for sample_path in rep_path.glob('*.pdbqt')),
                                              key=result_sorter)
                    else:
                        state_paths = sorted((state_path for state_path in rep_path.iterdir()),
                                             key=rip_last_number)
                        result_paths = [sorted((result_path for result_path in state_path.iterdir()),
                                               key=result_sorter)
                                        for state_path in state_paths]
                    if args.save_Kd != True:
                        extracted_results = pool.map(extractor, result_paths)
                    else:
                        extracted_results_Kd = pool.map(extractor, result_paths, args.save_Kd)
                        Kd = [row[1] for row in extracted_results_Kd]
                        extracted_results = [row[0] for row in extracted_results_Kd]
                        print(f'extracted_results:\n{Kd}')
                        rep_dir = out_path / rep_path.stem
                        rep_dir.mkdir(exist_ok=True, parents=True)
                        outpre = rep_dir / ligand_path.stem
                        r = ra.RaggedArray(Kd, error_checking=False)
                        ra.save(f'{outpre}_Kd.h5',r)

                    rep_dir = out_path / rep_path.stem
                    rep_dir.mkdir(exist_ok=True, parents=True)
                    outpre = rep_dir / ligand_path.stem
                    r = ra.RaggedArray(extracted_results)
                    ra.save(str(outpre.with_suffix('.h5')), r)
                    with outpre.with_suffix('.pickle').open('wb') as f:
                        pickle.dump(result_paths, f)
            else:
                if args.centers or args.centers_tag:
                    result_paths = sorted(([state_path] for state_path in ligand_path.glob('*.pdbqt')),
                                          key=result_sorter)
                else:
                    state_paths = sorted((state_path for state_path in ligand_path.iterdir()),
                                         key=lambda x: int(x.stem))
                    if args.result_type != 'gnina':
                        result_paths = [sorted((sample_path for sample_path in state_path.glob('*.pdbqt')),
                                           key=result_sorter)
                                    for state_path in state_paths]
                    else:
                        result_paths = [sorted((sample_path for sample_path in state_path.glob('*.sdf')),
                                           key=result_sorter)
                                    for state_path in state_paths]
                if args.save_Kd != True:
                    extracted_results = pool.map(extractor, result_paths)
                else:
                    extracted_results_Kd = pool.map(extractor, result_paths, args.save_Kd)
                    Kd = [row[1] for row in extracted_results_Kd]
                    extracted_results = [row[0] for row in extracted_results_Kd]
                    print(f'extracted_results:\n{Kd}')
                    outpre = out_path / ligand_path.stem
                    r = ra.RaggedArray(Kd, error_checking=False)
                    ra.save(f'{outpre}_Kd.h5',r)
                problem_states = []
                for i, result in enumerate(extracted_results):
                    if result.size == 0:  # no elements means some extration failed
                        problem_path = result_paths[i][0].parent
                        print(problem_path)
                        problem_states.append(problem_path)
                    if len(result) != 1:
                        print(result_paths[i], result)
                outpre = out_path / ligand_path.stem
                # Need to disable error checking in order to avoid a version bug issue.
                r = ra.RaggedArray(extracted_results, error_checking=False)
                if problem_states:
                    problems_out = (out_path / (ligand_path.stem + '-problems')).with_suffix('.txt')
                    print('There were issues extracting scores from some paths. See:', problems_out)
                    problems_out.write_text('\n'.join(map(str, problem_states)))
                else:
                    ra.save(str(outpre.with_suffix('.h5')), r)
                    with outpre.with_suffix('.pickle').open('wb') as f:
                        pickle.dump(result_paths, f)

pool.close()

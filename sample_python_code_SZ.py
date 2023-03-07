#!/usr/bin/env python
# coding: utf-8

# # Creating the list of atoms/bonds/angles/dihedrals and their corresponding types in kerogen models that are produced with many-body potentials such as AIREBO and REAXFF. The produced files can be used with DREIDING force field. 

# # This code is optimized for large models of kerogen (> 10k atoms) through vecetorization. It creates a list of bonds/dihedrals/angles for EFK kerogen model (10k atoms) in 24 minutes .

# In[54]:


import numpy as np
import pandas as pd
from numpy.linalg import norm


# In[60]:


# Filename containing atom coordinates in LAMMPS format (atom_style: FULL)
fileatom = 'atoms_efk_only_full_super.txt'


# In[332]:


data_atom = np.loadtxt(fileatom,skiprows=2)
#data_bonds = np.loadtxt(filebonds,skiprows=2)
#data_angles = np.loadtxt(fileangles,skiprows=2)


# cell sizes

# In[58]:


xlo=-0.15913034
ylo=-0.16106295
zlo=-0.15960394
xhi=23.47773034
yhi=143.38249245
zhi=117.00000000


# In[333]:


# Store atom coordinates into a dataframe
atoms=coordinates_to_df(data_atom)


# In[334]:


vec_a,vec_b,vec_c=cell_vectors(xlo,xhi,ylo,yhi,zlo,zhi)


# In[432]:


get_ipython().run_cell_magic('time', '', '## find bonds in periodic cell-- limited to kerogen (organic matter) systems in shale\nlist_all_bonds,list_all_bdist=find_bonds(atoms,vec_a,vec_b,vec_c)\n## Atom typing\natom_types(list_all_bonds,atoms)\n## Bond typing\nall_bond_type=assign_bond_type(list_all_bonds,atoms)\n## Write atom coordinates, bonds, angles, dihedral to files suitable for LAMMPS input\nbond_unique=make_lmp_databonds(all_bond_type,list_all_bonds,atoms)\nmake_lmp_dataangles_fast(bond_unique,atoms)\nmake_lmp_datadihedrals_fast(bond_unique,atoms)\nmake_lmp_atoms(atoms)')


# Atom typing... We find hybridization of carbon and oxygen atoms in the structure based on bonds



# In[326]:


def atom_types(list_all_bonds,atoms):
    
    total_charge=0
    #atoms["DREIDING TYPE"] = ""
    for idx, atom1 in atoms.iterrows():
      #  if atoms1['ATOM TYPE']==1:
         #   total_charge=4
            total_attached=len(list_all_bonds[idx])
            if total_attached==4 and atom1['ATOM TYPE']==1: #sp3
                atoms.at[idx,'DREIDING TYPE']=3
            elif  total_attached==3 and atom1['ATOM TYPE']==1: #sp2
                atoms.at[idx,'DREIDING TYPE']=2
            elif  total_attached==2 and atom1['ATOM TYPE']==1: #sp
                atoms.at[idx,'DREIDING TYPE']=1
            elif atom1['ATOM TYPE']==3 and total_attached==2: # sp3 oxygen
                atoms.at[idx,'DREIDING TYPE']=5
            elif atom1['ATOM TYPE']==3 and total_attached==1:   #sp2 oxygen
                atoms.at[idx,'DREIDING TYPE']=4
            elif atom1['ATOM TYPE']==2:   # hydrogen
                atoms.at[idx,'DREIDING TYPE']=6
            else:
                atoms.at[idx,'DREIDING TYPE']=5
    #atoms['DREIDING TYPE']=  atoms['DREIDING TYPE'].astype('int')
    atoms[atoms['DREIDING TYPE'].isnull()]
    #atoms['DREIDING TYPE']=atoms['DREIDING TYPE'].astype('int')
    #atoms


# Assigning the bond types

# In[308]:


def coordinates_to_df(data_atom):
    atoms = pd.DataFrame(data_atom, columns = ['ATOM ID','MOLECULE ID','ATOM TYPE','ATOM CHARGE','COORDX','COORDY','COORDZ','NOT USED','NOT USED','NOT USED'])
    atoms['ATOM ID']=atoms['ATOM ID'].astype('int')
    atoms['MOLECULE ID']=atoms['ATOM ID'].astype('int')
    atoms['ATOM TYPE']=atoms['ATOM TYPE'].astype('int')
    return atoms
    


# In[310]:


##Defining the coefficient matrix for periodic boundary conditions. This matrix will be later used to find the minimum image distance.
def pbc_coeff():
    cnt=0
    coeff=[]
    for i in range (-1,2):
        a1 = i
        for j in range (-1,2):
            b1=j
            for k in range (-1,2):
                c1=k
                coeff.append([a1 ,b1, c1])
                cnt=cnt+1
    coeff=np.array(coeff)

    return coeff


# In[311]:


# Putting cell dimensions (lo,hi) into vectors
def cell_vectors(xlo,xhi,ylo,yhi,zlo,zhi):
    
    vec_a=np.array([xhi-xlo ,0, 0])
    vec_b=np.array([0 ,yhi-ylo, 0])
    vec_c=np.array([0 ,0, zhi-zlo])
    return vec_a,vec_b,vec_c


# In[426]:


# Find bonds in kerogen (organic matter in shale) in a periodic cell
def find_bonds(atoms,vec_a,vec_b,vec_c):
    
    coeff=pbc_coeff()

    list_all_bonds=[]

    for index1, atom1 in atoms.iterrows():
        #print(atom1)
        print(index1)
        bonded_list=[]
        bond_dist_list=[]
        coord1 = atom1[['COORDX','COORDY','COORDZ']]
        # Coeff is used to calculate all the 27 distances between all image cells in 3D. Vectorization is done here through the dot product for distance vectors: 
        innerp=[np.sum((coord1-atoms[['COORDX','COORDY','COORDZ']]+coeff[k,0]*vec_a+coeff[k,1]*vec_b+coeff[k,2]*vec_c)*(coord1-atoms[['COORDX','COORDY','COORDZ']]+coeff[k,0]*vec_a+coeff[k,1]*vec_b+coeff[k,2]*vec_c),axis=1) for k in range(0,27)]
        min_innerp=np.amin(innerp,axis=0)
        dist12= np.sqrt(min_innerp)
        # Finding bonds based on bond threshold(1.3 for bonds invloving hydrogen, and 1.9 for the rest. These values are roughly based on bonds formed with REAXFF)
        #bonded = [idx for idx,element in enumerate(dist12) if (((atom1['ATOM TYPE']==2 or atoms.iloc[idx]['ATOM TYPE']==2) and dist12[idx]!=0 and dist12[idx] < 1.3) or (atom1['ATOM TYPE']!=2 and atoms.iloc[idx]['ATOM TYPE']!=2 and dist12[idx]!=0 and dist12[idx]< 1.9))]
        bonded=atoms.index[((dist12 < 1.3) & (dist12 > 0) & ((atom1['ATOM TYPE']==2) | (atoms['ATOM TYPE']==2))) | ((dist12 < 1.9) & (dist12 > 0) & ((atom1['ATOM TYPE']!=2) & (atoms['ATOM TYPE']!=2)))].tolist()
       # if index1 == 0: print(dist12)
       # if atom1['ATOM TYPE']==2 or atom2['ATOM TYPE']==2:
        #     bond_cutoff= 1.3
        #else:
        #     bond_cutoff= 1.9
        #if dist12 < bond_cutoff and dist12 != 0:
      #  bonded_list.append(bonded)
      #  bond_dist_list.append(dist12[bonded])
                 #if bond_type
            #    min_dist=min(dist)
        list_all_bonds.append(bonded) 
        list_all_bdist.append(dist12[bonded])
    return list_all_bonds,list_all_bdist
            #if (index1 == 1203 & index2 == 1207):
              #  print(dist12)
       # if index1 == 1: break


    


# In[329]:


def assign_bond_type(list_all_bonds,atoms):
    databond=pd.DataFrame({'BOND ID' : []})
    cnt=0
    all_bond_type=[]
    for idx_i, bondlist_i in enumerate(list_all_bonds):
      #  if idx_i==796:
      #      print("target atom: 796")
       #     print("target atom bondnum: 796",len(bondlist_i))
         #   total_charge=4
        bond_num=len(bondlist_i)
        bond_type=[]
        if bond_num==4 and (all(atoms.iloc[list_all_bonds[idx_i][id_j]]['ATOM TYPE']==1 for id_j in range(0,bond_num))):  
            bond_type.extend([1, 1 ,1, 1])
        elif bond_num==2 and atoms.iloc[idx_i]['ATOM TYPE']==3:
            bond_type.extend([2 ,2]) # need to be edited later for the case of O-H bonds
        elif bond_num==1 and atoms.iloc[idx_i]['ATOM TYPE']==2:
            bond_type.extend([3])
        else:
            for idx_j, bondij in enumerate(bondlist_i):
                if list_all_bdist[idx_i][idx_j] < 1.3 and atoms.iloc[list_all_bonds[idx_i][idx_j]]['ATOM TYPE']==2:
                    bond_type.extend([3])
                    if idx_i==796:
                        print("target atom: 796",1)
                elif list_all_bdist[idx_i][idx_j] < 1.3 and atoms.iloc[list_all_bonds[idx_i][idx_j]]['ATOM TYPE']==1:
                    bond_type.extend([4])
                    if idx_i==796:
                        print("target atom: 796",2)
                elif list_all_bdist[idx_i][idx_j] > 1.3 and list_all_bdist[idx_i][idx_j] < 1.6 and atoms.iloc[list_all_bonds[idx_i][idx_j]]['ATOM TYPE']==3:
                    bond_type.extend([2])
                    if idx_i==796:
                        print("target atom: 796",3)
                elif list_all_bdist[idx_i][idx_j] > 1.3 and list_all_bdist[idx_i][idx_j] < 1.6 and atoms.iloc[list_all_bonds[idx_i][idx_j]]['ATOM TYPE']==1:
                    bond_type.extend([1])
                    if idx_i==796:
                        print("target atom: 796",4)
                elif list_all_bdist[idx_i][idx_j] > 1.6:
                    bond_type.extend([1])
                    if idx_i==796:
                        print("target atom: 796",5)
                else:
                    bond_type.extend([2])
        all_bond_type.append(bond_type)
    return all_bond_type


# In[31]:

# Writing the list of bond to file suitable for LAMMPS input

def make_lmp_databonds(all_bond_type,list_all_bonds,atoms):
    
    f = open("lmp_bond_data.txt", "w")
    f.write("Bonds\n")
    f.write("\n")
    cnt=0
    bond_unique=[]
    # Creates the bond structure suitable for lammps 
    for idx_i, bondlist_i in enumerate(list_all_bonds):
        print(bondlist_i)
        for idx_j, bondij in enumerate(bondlist_i):

            if [bondij,idx_i] not in bond_unique:
                print("OK")
                cnt+=1
                bond_unique.append([idx_i,bondij])
                f.write("%s %s %s %s \n"%(cnt,all_bond_type[idx_i][idx_j],int(atoms.iloc[idx_i]['ATOM ID']),int(atoms.iloc[bondij]['ATOM ID'])))
        if cnt==3: break
    f.close()
    return bond_unique


# In[375]:

# Writing the list of angles to file suitable for LAMMPS input
def make_lmp_dataangles_fast(bond_unique,atoms):
    # FInd angles
    f = open("lmp_angle_data.txt", "w")
    f.write("Angles\n")
    f.write("\n")
    cnt=0
    bond_list_tmp=[]
    df_bonds=pd.DataFrame(bond_unique,columns=['ATOM I','ATOM J'])
    angles_unique=[]
   # bond_unique=[]

    #print(bond_js)
    # Loop over bonds to find angles. Vectorization (through shared_atom fuction) is used to accelrate the code.
    for idx_i,bond_i in df_bonds.iterrows():
              
        shatom=shared_atom(df_bonds,bond_i)
  
        for idx_k,bond_k in shatom.iterrows():
                 
            if bond_i['ATOM I']==bond_k['ATOM I']:
                shared_atom1=bond_i['ATOM I']
                notshared_atom1=bond_i['ATOM J']
                notshared_atom2=bond_k['ATOM J']
                angle=[shared_atom1,min(notshared_atom1,notshared_atom2),max(notshared_atom2,notshared_atom1)]
          #  shared_atom=shared_atom[0]
         #   if atoms.iloc[shared_atom]['DREIDING TYPE']
            elif bond_i['ATOM I']==bond_k['ATOM J']:
                shared_atom1=bond_i['ATOM I']
                notshared_atom1=bond_k['ATOM I']
                notshared_atom2=bond_i['ATOM J']
                angle=[shared_atom1,min(notshared_atom1,notshared_atom2),max(notshared_atom2,notshared_atom1)]
            elif bond_i['ATOM J']==bond_k['ATOM J']:
                shared_atom1=bond_i['ATOM J']
                notshared_atom1=bond_k['ATOM I']
                notshared_atom2=bond_i['ATOM I']
                angle=[shared_atom1,min(notshared_atom1,notshared_atom2),max(notshared_atom2,notshared_atom1)]
                #angle.sort()
            if not angle in bond_list_tmp:
                angles_unique.append(angle)
                cnt+=1
                # For DREIDING force field, equilibrium angle is based on the type of central atom. Stiffness depends on the type of the central atom
                if atoms.iloc[shared_atom1]['DREIDING TYPE']==3:
                    f.write("%s 3 %s %s %s \n"%(cnt,int(atoms.iloc[notshared_atom1]['ATOM ID']),int(atoms.iloc[shared_atom1]['ATOM ID']),int(atoms.iloc[notshared_atom2]['ATOM ID'])))
                elif atoms.iloc[shared_atom1]['DREIDING TYPE']==2:
                    f.write("%s 4 %s %s %s \n"%(cnt,int(atoms.iloc[notshared_atom1]['ATOM ID']),int(atoms.iloc[shared_atom1]['ATOM ID']),int(atoms.iloc[notshared_atom2]['ATOM ID'])))
                elif atoms.iloc[shared_atom1]['DREIDING TYPE']==1:
                    f.write("%s 5 %s %s %s \n"%(cnt,int(atoms.iloc[notshared_atom1]['ATOM ID']),int(atoms.iloc[shared_atom1]['ATOM ID']),int(atoms.iloc[notshared_atom2]['ATOM ID'])))
                elif atoms.iloc[shared_atom1]['DREIDING TYPE']==4:
                    f.write("%s 6 %s %s %s \n"%(cnt,int(atoms.iloc[notshared_atom1]['ATOM ID']),int(atoms.iloc[shared_atom1]['ATOM ID']),int(atoms.iloc[notshared_atom2]['ATOM ID'])))
                elif atoms.iloc[shared_atom1]['DREIDING TYPE']==5:
                    f.write("%s 7 %s %s %s \n"%(cnt,int(atoms.iloc[notshared_atom1]['ATOM ID']),int(atoms.iloc[shared_atom1]['ATOM ID']),int(atoms.iloc[notshared_atom2]['ATOM ID'])))
                print(cnt,atoms.iloc[shared_atom1]['ATOM ID'],atoms.iloc[notshared_atom1]['ATOM ID'],atoms.iloc[notshared_atom2]['ATOM ID'])
                bond_list_tmp.append(angle)
    f.close()


# In[339]:


def make_lmp_datadihedrals_fast(bond_unique,atoms):

    # Finding dihrdrals
    f = open("lmp_dihedral_data.txt", "w")
    f.write("Dihedrals\n")
    f.write("\n")
    cnt=0
    bond_list_tmp=[]
    df_bonds=pd.DataFrame(bond_unique,columns=['ATOM I','ATOM J'])

    
    # loop over bonds
    for idx_i,bond_i in df_bonds.iterrows():
        
        
        # Here, vectorization is performed to find bonds with shared atoms with bond_i
        shatom=shared_atom(df_bonds,bond_i)
  
        for idx_k,bond_k in shatom.iterrows():
            print(idx_i,idx_k)
            # Again, vectorization is performed to find bonds with shared atoms with bonds in shatom
            shatom2=shared_atom(df_bonds,bond_k)
            for idx_l, bond_l in shatom2.iterrows():
                
                
                if bond_i.tolist()!=bond_l.tolist():
                    
                    # finding shared atoms between the bonds in dihedral
                    shared_atom1=[bond_i.iloc[k]==bond_k.iloc[l] for k in range(0,2) for l in range(0,2)]
                    shared_atom2=[bond_i.iloc[k]==bond_l.iloc[l] for k in range(0,2) for l in range(0,2)]
                    shared_atom3=[bond_k.iloc[k]==bond_l.iloc[l] for k in range(0,2) for l in range(0,2)]
                    # Check if not all bonds share an atom!
                    if any(shared_atom1) and any(shared_atom2) and not any(shared_atom3):
                    
                        print(shared_atom1, shared_atom2, shared_atom3)
                        index1=shared_atom1.index(any(shared_atom1))
                        index2=shared_atom2.index(any(shared_atom2))
                        shared_atom1_val=bond_i[int(index1 > 1)]
                        shared_atom2_val=bond_i[int(index2 > 1)]
                        nshared_atom1_val=bond_k[1-index1 % 2]
                        nshared_atom2_val=bond_l[1-index2 % 2]
                        dihedral1=[nshared_atom1_val,shared_atom1_val,shared_atom2_val,nshared_atom2_val]
                        dihedral2=[nshared_atom2_val,shared_atom2_val,shared_atom1_val,nshared_atom1_val]
                        # Check if dihedral is not found yet
                        if not dihedral1 in bond_list_tmp and not dihedral2 in bond_list_tmp:
                            

                            cnt+=1
                            f.write("%s 1 %s %s %s %s\n"%(cnt,int(atoms.iloc[nshared_atom1_val]['ATOM ID']),int(atoms.iloc[shared_atom1_val]['ATOM ID']),int(atoms.iloc[shared_atom2_val]['ATOM ID']),int(atoms.iloc[nshared_atom2_val]['ATOM ID'])))
                            print(cnt,atoms.iloc[nshared_atom1_val]['ATOM ID'],atoms.iloc[shared_atom1_val]['ATOM ID'],atoms.iloc[shared_atom2_val]['ATOM ID'],atoms.iloc[nshared_atom2_val]['ATOM ID'])
                            bond_list_tmp.append(dihedral1)
                            bond_list_tmp.append(dihedral2)
                    elif any(shared_atom1) and any(shared_atom3) and not any(shared_atom2):
                        print(shared_atom1, shared_atom2, shared_atom3)
                        index1=shared_atom1.index(any(shared_atom1))
                        index2=shared_atom3.index(any(shared_atom3))
                        shared_atom1_val=bond_i[int(index1 > 1)]
                        shared_atom2_val=bond_k[int(index2 > 1)]
                        nshared_atom1_val=bond_i[1-int(index1 > 1)]
                        nshared_atom2_val=bond_l[1-index2 % 2]
                        dihedral1=[nshared_atom1_val,shared_atom1_val,shared_atom2_val,nshared_atom2_val]
                        dihedral2=[nshared_atom2_val,shared_atom2_val,shared_atom1_val,nshared_atom1_val]
                        if not dihedral1 in bond_list_tmp and not dihedral2 in bond_list_tmp:
                        
                            cnt+=1
                            f.write("%s 1 %s %s %s %s\n"%(cnt,int(atoms.iloc[nshared_atom1_val]['ATOM ID']),int(atoms.iloc[shared_atom1_val]['ATOM ID']),int(atoms.iloc[shared_atom2_val]['ATOM ID']),int(atoms.iloc[nshared_atom2_val]['ATOM ID'])))
                            print(22,cnt,atoms.iloc[nshared_atom1_val]['ATOM ID'],atoms.iloc[shared_atom1_val]['ATOM ID'],atoms.iloc[shared_atom2_val]['ATOM ID'],atoms.iloc[nshared_atom2_val]['ATOM ID'])
                            bond_list_tmp.append(dihedral1)
                            bond_list_tmp.append(dihedral2)
                    elif any(shared_atom2) and any(shared_atom3) and not any(shared_atom1):
                        print(shared_atom1, shared_atom2, shared_atom3)
                       # print(bond_i,bond_k,bond_l)
                        index1=shared_atom2.index(any(shared_atom2))
                        index2=shared_atom3.index(any(shared_atom3))
                        shared_atom1_val=bond_i[int(index1 > 1)]
                        shared_atom2_val=bond_k[int(index2 > 1)]
                        nshared_atom1_val=bond_i[1-int(index1 > 1)]
                        nshared_atom2_val=bond_k[1-int(index2 > 1)]
                        dihedral1=[nshared_atom1_val,shared_atom1_val,shared_atom2_val,nshared_atom2_val]
                        dihedral2=[nshared_atom2_val,shared_atom2_val,shared_atom1_val,nshared_atom1_val]
                        if not dihedral1 in bond_list_tmp and not dihedral2 in bond_list_tmp:
                            
                        
                            cnt+=1
                            f.write("%s 1 %s %s %s %s\n"%(cnt,int(atoms.iloc[nshared_atom1_val]['ATOM ID']),int(atoms.iloc[shared_atom1_val]['ATOM ID']),int(atoms.iloc[shared_atom2_val]['ATOM ID']),int(atoms.iloc[nshared_atom2_val]['ATOM ID'])))
                            print(23,cnt,atoms.iloc[nshared_atom1_val]['ATOM ID'],atoms.iloc[shared_atom1_val]['ATOM ID'],atoms.iloc[shared_atom2_val]['ATOM ID'],atoms.iloc[nshared_atom2_val]['ATOM ID'])
                            bond_list_tmp.append(dihedral1)
                            bond_list_tmp.append(dihedral2)
                     #   print(bond_list_tmp)
              #  shared_atom=shared_atom[0]
             #   if atoms.iloc[shared_atom]['DREIDING TYPE']
              
    f.close()


# In[340]:


def make_lmp_atoms(atoms):

    # Function written for lammps atom coordinates (atom style: full) suitable for DREIDING force field
    f = open("lmp_atoms_data.txt", "w")
    f.write("Atoms\n")
    f.write("\n")
    
    for idx_i,atom_i in atoms.iterrows():
        
        f.write("%s %s %s %s %s %s %s\n"%(int(atom_i['ATOM ID']),int(atom_i['MOLECULE ID']),int(atom_i['DREIDING TYPE']),atom_i['ATOM CHARGE'],atom_i['COORDX'],atom_i['COORDY'],atom_i['COORDZ']))
                                        
    f.close()


# In[371]:


def shared_atom(df_bonds,bondi):
    
 #   Vectorization to find bonds sharing atoms with bondi
    shatom=df_bonds.loc[((df_bonds['ATOM I']==bondi['ATOM I']) & (df_bonds['ATOM J']!=bondi['ATOM J'])) | ((df_bonds['ATOM J']==bondi['ATOM I']) & (df_bonds['ATOM I']!=bondi['ATOM J'])) | ((df_bonds['ATOM I']==bondi['ATOM J']) & (df_bonds['ATOM J']!=bondi['ATOM I']))  | ((df_bonds['ATOM J']==bondi['ATOM J']) & (df_bonds['ATOM I']!=bondi['ATOM I']))]
    return shatom


# In[327]:


# Function that returns the intersection of two sets
def intersection(set1,set2):
    
    intersect= [value for value in set1 if value in set2]
    return intersect


# In[328]:


# Function that returns the intersection of two sets
def intersectionconj(set1,set2):
    
    intersectconj= [value for value in set1 if value not in set2]
    return intersectconj


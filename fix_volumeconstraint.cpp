/* ----------------------------------------------------------------------
   Added by Yao Xiong at CCTSM of Northwestern Univeristy (January 20, 2023)
------------------------------------------------------------------------- */

#include "fix_volumeconstraint.h"

#include "atom.h"
#include "atom_masks.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "respa.h"
#include "update.h"
#include "variable.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NONE,CONSTANT,EQUAL,ATOM};

/* ---------------------------------------------------------------------- */

FixVolumeConstraint::FixVolumeConstraint(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), xstr(nullptr), ystr(nullptr)
{
  if (narg != 5) error->all(FLERR,"Illegal fix volumeconstraint command");

  dynamic_group_allow = 1;
  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  energy_global_flag = 1;
  respa_level_support = 1;
  ilevel_respa = 0;

  if (utils::strmatch(arg[3],"^v_")) {
    xstr = utils::strdup(arg[3]+2);
  } else {
    xvalue = utils::numeric(FLERR,arg[3],false,lmp);
    xstyle = CONSTANT;
  }
  if (utils::strmatch(arg[4],"^v_")) {
    ystr = utils::strdup(arg[4]+2);
  } else {
    yvalue = utils::numeric(FLERR,arg[4],false,lmp);
    ystyle = CONSTANT;
  }

  force_flag = 0;
  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;  
}

/* ---------------------------------------------------------------------- */

FixVolumeConstraint::~FixVolumeConstraint()
{
  delete [] xstr;
  delete [] ystr;
}

/* ---------------------------------------------------------------------- */

int FixVolumeConstraint::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixVolumeConstraint::init()
{
  // check variables

  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0)
      error->all(FLERR,"Variable name for fix volumeconstraint does not exist");
    if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
    else error->all(FLERR,"Variable for fix volumeconstraint is invalid style");
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0)
      error->all(FLERR,"Variable name for fix volumeconstraint does not exist");
    if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
    else error->all(FLERR,"Variable for fix volumeconstraint is invalid style");
  }

  if (xstyle == EQUAL || ystyle == EQUAL)
    varflag = EQUAL;
  else varflag = CONSTANT;

  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixVolumeConstraint::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style,"^verlet"))
    post_force(vflag);
  else {
    (dynamic_cast<Respa *>( update->integrate))->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag,ilevel_respa,0);
    (dynamic_cast<Respa *>( update->integrate))->copy_f_flevel(ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixVolumeConstraint::min_setup(int vflag)
{
    post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixVolumeConstraint::post_force(int /*vflag*/)
{
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  imageint *image = atom->image;
  int **trianglelist = neighbor->anglelist;
  int ntrianglelist = neighbor->nanglelist;
  int nlocal = atom->nlocal;

  int i1,i2,i3;
  double vecr1x,vecr1y,vecr1z,vecr2x,vecr2y,vecr2z,vecr3x,vecr3y,vecr3z;
  double dr1x,dr1y,dr1z,dr2x,dr2y,dr2z,dr3x,dr3y,dr3z;
  double f1[3],f2[3],f3[3];
  double unwrap[3];
  double vol,vcpref;
  tagint *tag = atom->tag;
  
  // foriginal[0] = "potential energy" for added force
  // foriginal[123] = force on atoms before extra force added
  
  foriginal[0] = foriginal[1] = foriginal[2] = foriginal[3] = 0.0;
  force_flag = 0;

  if (varflag == EQUAL) {
    modify->clearstep_compute();
    if (xstyle == EQUAL) xvalue = input->variable->compute_equal(xvar);
    if (ystyle == EQUAL) yvalue = input->variable->compute_equal(yvar);
    modify->addstep_compute(update->ntimestep + 1);
  }
  
  vol = 0.0;
  
  for (int i = 0; i < ntrianglelist; i++) {
    i1 = trianglelist[i][0];
    i2 = trianglelist[i][1];
    i3 = trianglelist[i][2];
    
    // 1st vertice of the triangle
    domain->unmap(x[i1],image[i1],unwrap);
      
    vecr1x = unwrap[0];
    vecr1y = unwrap[1];
    vecr1z = unwrap[2];
      
    // 2nd vertice of the triangle
    domain->unmap(x[i2],image[i2],unwrap);
      
    vecr2x = unwrap[0];
    vecr2y = unwrap[1];
    vecr2z = unwrap[2];
      
    // 3rd vertice of the triangle
    domain->unmap(x[i3],image[i3],unwrap);
      
    vecr3x = unwrap[0];
    vecr3y = unwrap[1];
    vecr3z = unwrap[2]; 
        
    // calculate the volume
    // vol = Sum(1/6 * r1 . (r2 x r3) =  (-x3 y2 + x2 y3) z1 + (x3 z2 - x2 z3) y1 + (-y3 z2 + y2 z3) x1)
    /*fmt::print(screen,"1st atom : {}\n",i1);
    fmt::print(screen,"2nd atom : {}\n",i2);
    fmt::print(screen,"3rd atom : {}\n",i3);*/
    /*fmt::print(screen,"vc1: {} {} {} {} {}\n",tag[i1],vecr1x,vecr1y,vecr1z,image[i1]);
    fmt::print(screen,"vc1: {} {} {} {} {}\n",tag[i2],vecr2x,vecr2y,vecr2z,image[i2]);
    fmt::print(screen,"vc1: {} {} {} {} {}\n",tag[i3],vecr3x,vecr3y,vecr3z,image[i3]);*/
      
      /*fmt::print(screen,"Triangle No. {}\n",i);
      fmt::print(screen,"1st {} {} {} 2nd {} {} {} 3rd {} {} {}\n",vecr1x,vecr1y,vecr1z,vecr2x,vecr2y,vecr2z,vecr3x,vecr3y,vecr3z);*/
      
      /*if ((vecr1y > 20.0) || (vecr2y > 20.0) || (vecr3y > 20.0))  {
        fmt::print(screen,"1st atom : {} {} {}\n",vecr1x,vecr1y,vecr1z);
        fmt::print(screen,"2nd atom : {} {} {}\n",vecr2x,vecr2y,vecr2z);
        fmt::print(screen,"3rd atom : {} {} {}\n",vecr3x,vecr3y,vecr3z);
      }*/
      
      /*fmt::print(screen,"IDs: {} {} {}\n",tag[i1],tag[i2],tag[i3]);*/
    vol += 1.0 / 6.0 * ((-vecr3x * vecr2y + vecr2x * vecr3y) * vecr1z + (vecr3x * vecr2z - vecr2x * vecr3z) * vecr1y + (-vecr3y * vecr2z + vecr2y * vecr3z) * vecr1x);
  }
  
  /*fmt::print(screen,"the volume : {}\n",vol);*/
  /*fmt::print(screen,"1st atom : {} {} {}\n",vecr1x,vecr1y,vecr1z);
  fmt::print(screen,"2nd atom : {} {} {}\n",vecr2x,vecr2y,vecr2z);
  fmt::print(screen,"3rd atom : {} {} {}\n",vecr3x,vecr3y,vecr3z);*/
  
  // foriginal[0] = "potential energy" for added force
  // foriginal[123] = force on atoms before extra force added
  // potential energy = lambda * (vol - vol_ref) ^ 2
    
  foriginal[0] = xvalue * pow((vol - yvalue),2);
   
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      foriginal[1] += f[i][0];
      foriginal[2] += f[i][1];
      foriginal[3] += f[i][2];
    }
    
  for (int j = 0; j < ntrianglelist; j++) {
    i1 = trianglelist[j][0];
    i2 = trianglelist[j][1];
    i3 = trianglelist[j][2];
    
    // 1st vertice of the triangle
    domain->unmap(x[i1],image[i1],unwrap);
      
    vecr1x = unwrap[0];
    vecr1y = unwrap[1];
    vecr1z = unwrap[2];
      
    // 2nd vertice of the triangle
    domain->unmap(x[i2],image[i2],unwrap);
      
    vecr2x = unwrap[0];
    vecr2y = unwrap[1];
    vecr2z = unwrap[2];
      
    // 3rd vertice of the triangle
    domain->unmap(x[i3],image[i3],unwrap);
      
    vecr3x = unwrap[0];
    vecr3y = unwrap[1];
    vecr3z = unwrap[2];
    
    /*fmt::print(screen,"vc2: {} {} {} {} {}\n",tag[i1],r1x,r1y,r1z,image[i1]);
    fmt::print(screen,"vc2: {} {} {} {} {}\n",tag[i2],r2x,r2y,r2z,image[i2]);
    fmt::print(screen,"vc2: {} {} {} {} {}\n",tag[i3],r3x,r3y,r3z,image[i3]);*/
      
    // on the paper, the formula turns out to be: f_i = 2*Lambda*Sum(1/6 * D[r1 . (r2 x r3), r1])
    // D[r1 . (r2 x r3), vec1] = (-y3*z2+y2*z3, x3*z2-x2*z3, -x3*y2+x2*y3)
    // the summation is over all triangles that contain vertice i
      
    dr1x = -vecr3y * vecr2z + vecr2y * vecr3z;
    dr1y = vecr3x * vecr2z - vecr2x * vecr3z;
    dr1z = -vecr3x * vecr2y + vecr2x * vecr3y;
      
    // D[r1 . (r2 x r3), r2] = (y3*z1-y1*z3, -x3*z1+x1*z3, x3*y1-x1*y3)
      
    dr2x = vecr3y * vecr1z - vecr1y * vecr3z;
    dr2y = -vecr3x * vecr1z + vecr1x * vecr3z;
    dr2z = vecr3x * vecr1y - vecr1x * vecr3y;
      
    // D[r1 . (r2 x r3), r3] = (-y2*z1+y1*z2, x2*z1-x1*z2, -x2*y1+x1*y2)
      
    dr3x = -vecr2y * vecr1z + vecr1y * vecr2z;
    dr3y = vecr2x * vecr1z - vecr1x * vecr2z;
    dr3z = -vecr2x * vecr1y + vecr1x * vecr2y;
      
    // actually calculate the force
    // compute the force for each atom in the triangle
      
    vcpref = -2.0 * xvalue / 6.0;
      
    f1[0] = vcpref * dr1x * (vol - yvalue);
    f1[1] = vcpref * dr1y * (vol - yvalue);
    f1[2] = vcpref * dr1z * (vol - yvalue);
      
    f2[0] = vcpref * dr2x * (vol - yvalue);
    f2[1] = vcpref * dr2y * (vol - yvalue);
    f2[2] = vcpref * dr2z * (vol - yvalue);
      
    f3[0] = vcpref * dr3x * (vol - yvalue);
    f3[1] = vcpref * dr3y * (vol - yvalue);
    f3[2] = vcpref * dr3z * (vol - yvalue);
      
    f[i1][0] += f1[0];
    f[i1][1] += f1[1];
    f[i1][2] += f1[2];
      
    f[i2][0] += f2[0];
    f[i2][1] += f2[1];
    f[i2][2] += f2[2];
      
    f[i3][0] += f3[0];
    f[i3][1] += f3[1];
    f[i3][2] += f3[2];
  }
}

/* ---------------------------------------------------------------------- */

void FixVolumeConstraint::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixVolumeConstraint::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   potential energy of added torque
------------------------------------------------------------------------- */

double FixVolumeConstraint::compute_scalar()
{
  // only sum across procs one time

  if (force_flag == 0) {
    MPI_Allreduce(foriginal, foriginal_all, 4, MPI_DOUBLE, MPI_SUM, world);
    force_flag = 1;
  }
  return foriginal_all[0];
}

/* ----------------------------------------------------------------------
   return components of total force on fix group before force was changed
------------------------------------------------------------------------- */

double FixVolumeConstraint::compute_vector(int n)
{
  // only sum across procs one time

  if (force_flag == 0) {
    MPI_Allreduce(foriginal, foriginal_all, 4, MPI_DOUBLE, MPI_SUM, world);
    force_flag = 1;
  }
  return foriginal_all[n + 1];
}

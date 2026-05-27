#include "fdtd_2d.h"

int main(double f0,
         int nt,
         int nx,
         int ny,
         double dx,
         int npml,
         double reflmax,
         int TFSF,
         int xante,
         char mode,
         double **ne,
         double **b0,
         double *amplitud,
         double *phase,
         double **ampl_inc,
         double **phase_inc,
         double **phase_ref,
         int n_recv,
         int *yrecv,
         int recv_width,
         double *ampl_recv,
         double *fase_recv,
         _Bool save_diag,
         char * outp_dir
       ){

  struct inputdata data;

  data.f0 = f0;
  data.nt = nt;
  data.nx = nx;
  data.ny = ny;
  data.dx = dx;
  data.npml = npml;
  data.reflmax = reflmax;
  data.TFSF = TFSF;
  data.xante = xante;

  data.ne = ne;

  data.ampl_ant = amplitud;
  data.fase_ant = phase;

  data.ampl_inc = ampl_inc;
  data.phase_inc = phase_inc;
  data.phase_ref = phase_ref;

  /* PCR receiver array */
  data.n_recv    = n_recv;
  data.yrecv     = yrecv;
  data.recv_width = recv_width;
  data.ampl_recv = ampl_recv;
  data.fase_recv = fase_recv;

  data.save_diag = save_diag;
  data.outp_dir = outp_dir;

  if (mode == 'O'){
    printf("Starting O-mode 2DFW simulation.\n");
    maxwell_2d_omode (&data);
  }
  else if (mode == 'X'){
    data.b0 = b0;
  printf("Starting X-mode 2DFW simulation.\n");
    maxwell_2d_xmode (&data);
  }
  else
    printf("Invalid polarization mode, must be 'X' or 'O'.");

  return 0;
}

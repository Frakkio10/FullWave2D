/* Velocidad de la luz en vacio (m/s) */
#define C 2.99792458e8

/* Carga del electron (C) */
#define E 1.602176487e-19

/* Masa del electron (kg) */
#define ME 9.10938215e-31

/* Permitividad del espacio libre (F/m) */
#define E0 8.854187817e-12

/* Permeabilidad magnetica del espacio libre (H/m) */
#define U0 1.2566370614e-6

/* Impedancia del espacio libre (ohm) */
#define Z0 376.730313461

/* Numero PI */
#define PI 3.141592653589

/* Factor de estabilidad de Courant */
#define S 0.5

#define FALSE 0

#define TRUE !(FALSE)

#include <stdio.h>


struct inputdata
{
  double f0;        /* Frecuencia onda incidente (Hz) */
  int nt;           /* Numero iteraciones temporales */
  int nx;           /* Numero de puntos de malla eje X*/
  int ny;           /* numero de puntos de malla eje Y */
  double dx;        /* Resolucion espacial de la malla */

  int npml;
  double reflmax;
  int TFSF;
  int xante;

  double **ne;      /* Densidad del plasma */
  double **b0;      /* magnetic field of the device */
  double *ampl_ant; /* Amplitud en la antena */
  double *fase_ant; /* Fase en la antena */
  double **ampl_inc;
  double **phase_inc;

  /* PCR receiver array */
  int     n_recv;       /* number of receivers (0 = DBS mode) */
  int    *yrecv;        /* poloidal index of each receiver [n_recv] */
  int     recv_width;   /* half-width of collection region (grid points) */
  double *ampl_recv;    /* output amplitude per receiver [n_recv] */
  double *fase_recv;    /* output phase per receiver [n_recv] */

  _Bool save_diag;
  char *outp_dir;
};

static void set_sigma (double **sigma, int nrows, int ncols, double dx, int n, int npml, double reflmax, char ax[1], _Bool star, char mode[1]) {
  int i, j, k;
  double a;
  int b;
  double sigmamax = -3.0*log(reflmax)/(2.0*Z0*npml*dx);

  if (star){
    sigmamax = sigmamax * Z0*Z0;
  }

  if ((mode == 'O' && star) || (mode == 'X' && !star)){
    a = 0.5;
    b = -1;
  }
  else {
    a = 0;
    b = 0;
  }

  for (j = 0; j <= nrows - 1; j++) {
    for (i = 0; i <= ncols - 1; i++) {

      if (ax == 'x') k = i;
      else if (ax == 'y') k = j;

      if (k <= npml + b)
      sigma[j][i] = sigmamax*pow((double)(k + a - npml)/(double)(npml), 2.0);
      else if (k >= n-npml)
      sigma[j][i] = sigmamax*pow((double)(k + a - (n-npml))/(double)(npml), 2.0);
      else
      sigma[j][i] = 0.0;
    }
  }
}

static double **memory (int filas, int cols) {
  /* Reserva memoria para alojar un array bi-dimensional con 'filas' */
  /* filas y 'cols' columnas. Devuelve un puntero a puntero.Cada     */
  /* elemento del array de punteros apunta a cada una de las filas   */
  /* del array bi-dimensional                                        */
  /* Ademas inicializa el array bidimensional con ceros              */
  int i, j;
  double **pf;

  /* Reserva memoria para el array de punteros */
  if ((pf = (double **) malloc (filas * sizeof (double *)))== NULL) {
    printf("Insuficient memory\n");
    exit(-1);
  }

  /* Reserva memoria para cada fila con cols columnas */
  for (j = 0; j < filas; j ++)
    if ((pf[j] = (double *)malloc (cols * sizeof(double))) == NULL) {
      printf("Insuficient memory. Exit\n");
      exit(-1);
    }

  /* Inicializa el array bidimensional a cero */
  for (j = 0; j < filas; j++)
    for (i = 0; i < cols; i++)
      pf[j][i] = 0.0;

  return pf;

};

static void memory_free (double **pf, int filas) {

  int j;
  for (j = 0; j < filas; j++) {
    free(pf[j]);
    pf[j] = NULL;
  }
  free (pf);
};

static void save_2d_arr_to_file(int nx, int ny, double **arr, FILE *f){
  int i,j;
  for (j = ny; j >= 0; j--){
      for (i = 0; i <= nx; i++){
        fprintf(f, " %f", arr[j][i]);
      }
      fprintf(f, "\n");
    }
};

// static void save_2d_arr_compressed(int nx, int ny, double **arr, int ncompr, FILE *f){
//   int i,j;
//   for (j = ny/ncompr; j >= 0; j--){
//       for (i = 0; i <= nx/ncompr; i++){
//         fprintf(f, " %f", arr[j*ncompr][i*ncompr]);
//       }
//       fprintf(f, "\n");
//     }
// };


int maxwell_2d_omode (struct inputdata *data);

int maxwell_2d_xmode (struct inputdata *data);

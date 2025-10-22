struct inputdata
{
  double f0;        /* Frecuencia onda incidente (Hz) */
  int nt;           /* Numero iteraciones temporales */
  int nx;           /* Numero de puntos de malla eje X*/
  int ny;           /* numero de puntos de malla eje Y */
  double dx;        /* Resolucion espacial de la malla */
  int yante;        /* Posicion vertical de la antena */
  int waist;        /* Beam waist en puntos de malla */
  double angle;     /* Angulo de propagacion */
  double **ne;      /* Densidad del plasma */
  double **b0;      /* Campo magnetico */
  double *ampl_ant; /* Amplitud en la antena */
  double *fase_ant; /* Fase en la antena */
  double nfact; /* factor for density (LV) */

};


int maxwell_2d_xmode (struct inputdata *data);


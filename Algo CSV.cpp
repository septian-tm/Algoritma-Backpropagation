#include "stdio.h"
#include "math.h"
#include "stdlib.h"

#define MAX_ROWS 100
#define MAX_COLS 3
//Perintah untuk membaca csv
void read_csv(const char *filename, float x[MAX_ROWS][2], float t[MAX_ROWS], int *rows) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Tidak dapat membuka file %s\n", filename);
        exit(1);
    }

    char line[256];
    int row = 0;
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "%f,%f,%f", &x[row][0], &x[row][1], &t[row]) == 3) {
            row++;
        }
    }
    fclose(file);

    *rows = row;
}

int main() {
    int max_iter;
    printf("Masukkan jumlah iterasi maksimum: ");
    scanf("%d", &max_iter);

    int baris;
    int kolom = 2;
    int jumlah_hidden_units = 3;
    int jumlah_output_units = 1;

    float x[MAX_ROWS][2];
    float t[MAX_ROWS];
    float b = 1 ;

    // Membaca data dari file CSV
    read_csv("D:/Algoritma/CSV_BP.csv", x, t, &baris);

    float v[jumlah_hidden_units][kolom + 1];
    float w[jumlah_hidden_units + 1];

    for (int baris_v = 0; baris_v < jumlah_hidden_units; baris_v++) {
        for (int kolom_v = 0; kolom_v < kolom + 1; kolom_v++) {
            v[baris_v][kolom_v] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        }
    }

    for (int kolom_w = 0; kolom_w < jumlah_hidden_units + 1; kolom_w++) {
        w[kolom_w] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
    // Data Training
    float dv[jumlah_hidden_units][kolom + 1], dw[jumlah_hidden_units + 1];
    float e = 2.718;
    float alpha = 0.8;
    float z_in[jumlah_hidden_units], z[jumlah_hidden_units];
    float y_in, y;

    float teta;

    int loop = 0;

    printf("Bobot awal:\n");
    for (int baris_v = 0; baris_v < jumlah_hidden_units; baris_v++) {
        for (int kolom_v = 0; kolom_v < kolom + 1; kolom_v++) {
            printf("v[%d][%d]: %.3f ", baris_v, kolom_v, v[baris_v][kolom_v]);
        }
        printf("\n");
    }
    for (int kolom_w = 0; kolom_w < jumlah_hidden_units + 1; kolom_w++) {
        printf("w[%d]: %.3f ", kolom_w, w[kolom_w]);
    }

    while (loop < max_iter) {
        for (int baris_x = 0; baris_x < baris; baris_x++) {
            for (int kolom_v = 0; kolom_v < jumlah_hidden_units; kolom_v++) {
                z_in[kolom_v] = b * v[kolom_v][0];
                for (int kolom_x = 0; kolom_x < kolom; kolom_x++) {
                    z_in[kolom_v] += x[baris_x][kolom_x] * v[kolom_v][kolom_x + 1];
                }
                z[kolom_v] = 1 / (1 + pow(e, -z_in[kolom_v]));
            }

            y_in = b * w[0];
            for (int kolom_z = 0; kolom_z < jumlah_hidden_units; kolom_z++) {
                y_in += z[kolom_z] * w[kolom_z + 1];
            }
            y = 1 / (1 + pow(e, -y_in));

            teta = (t[baris_x] - y) * y * (1 - y);

            for (int kolom_dw = 0; kolom_dw < jumlah_hidden_units + 1; kolom_dw++) {
                dw[kolom_dw] = alpha * teta * (kolom_dw == 0 ? b : z[kolom_dw - 1]);
            }

            float t_in[jumlah_hidden_units], t_1[jumlah_hidden_units];
            for (int kolom_v = 0; kolom_v < jumlah_hidden_units; kolom_v++) {
                t_in[kolom_v] = teta * w[kolom_v + 1];
                t_1[kolom_v] = t_in[kolom_v] * z[kolom_v] * (1 - z[kolom_v]);
                for (int kolom_dv = 0; kolom_dv < kolom + 1; kolom_dv++) {
                    dv[kolom_v][kolom_dv] = alpha * t_1[kolom_v] * (kolom_dv == 0 ? b : x[baris_x][kolom_dv - 1]);
                }
            }

            for (int kolom_w = 0; kolom_w < jumlah_hidden_units + 1; kolom_w++) {
                w[kolom_w] += dw[kolom_w];
            }

            for (int baris_v = 0; baris_v < jumlah_hidden_units; baris_v++) {
                for (int kolom_v = 0; kolom_v < kolom + 1; kolom_v++) {
                    v[baris_v][kolom_v] += dv[baris_v][kolom_v];
                }
            }
        }
        loop++;
    }
    //Bagian Testing
    printf("\n\nBobot akhir setelah proses training algoritma Backpropagation (%i epoch):\n", loop);
    for (int baris_v = 0; baris_v < jumlah_hidden_units; baris_v++) {
        for (int kolom_v = 0; kolom_v < kolom + 1; kolom_v++) {
            printf("v[%d][%d]: %.3f ", baris_v, kolom_v, v[baris_v][kolom_v]);
        }
        printf("\n");
    }

    printf("w: ");
    for (int kolom_w = 0; kolom_w < jumlah_hidden_units + 1; kolom_w++) {
        printf("%.3f ", w[kolom_w]);
    }

    printf("\n\nBagian Testing\n\n");
    printf(" x1 x2  b  y\n");
    printf("------------\n");

    for (int baris_x = 0; baris_x < baris; baris_x++) {
        for (int kolom_v = 0; kolom_v < jumlah_hidden_units; kolom_v++) {
            z_in[kolom_v] = b * v[kolom_v][0];
            for (int kolom_x = 0; kolom_x < kolom; kolom_x++) {
                z_in[kolom_v] += x[baris_x][kolom_x] * v[kolom_v][kolom_x + 1];
            }
            z[kolom_v] = 1 / (1 + pow(e, -z_in[kolom_v]));
        }

        y_in = b * w[0];
        for (int kolom_z = 0; kolom_z < jumlah_hidden_units; kolom_z++) {
            y_in += z[kolom_z] * w[kolom_z + 1];
        }
        y = 1 / (1 + pow(e, -y_in));

        printf(" %2.0f %2.0f %2.0f %2.0f\n", x[baris_x][0], x[baris_x][1], b, y);
    }

    printf("------------\n");

    return 0;
}

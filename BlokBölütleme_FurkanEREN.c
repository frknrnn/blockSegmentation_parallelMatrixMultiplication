#include <stdio.h>
#include <stdlib.h>
#include "mpi.h" //MPI kutuphanesi
//Furkan EREN 09.05.2020 Blok Bölütlenmiş Matris Matris Çarpımı
#define MASTER 0


float *create1DArray(int n) {
     float *T = (float *)malloc(n * sizeof(float));
     return T;
}

void fillArray(float *T, int n1) {
     int i;
     for (i = 0; i < n1; i++)
	T[i] = i+1.0; //i+1.0
}


float innerProd(float *u, float *v, int n) {
     int i;
     float sum = 0.0;
     for (i = 0; i < n; i++)
	  sum += u[i] * v[i];
     return sum;	
}

float *mat_vec_mult(float *M, float *v, int n1, int n2,int n3) {
     int f=0;
     float *r = create1DArray(n1*n3);
     for (int i = 0; i < n1; i++)
	for(int j=0; j < n3; j++)
	    r[f++] = innerProd(&M[i*n2], &v[j*n2], n2);
            
     return r;
}

float *change(float *M, int n1, int n2) {
     int f=0;
     float *r = create1DArray(n1*n2);
     for (int i = 0; i < n2; i++){
          float *b = &M[i];
	for(int j=0; j < n1; j++){


	    r[f++] =b[j*n2];
}	
}
    
     return r;
}


void printArray(float *T, int n1) {
     int i;
     for (i = 0; i < n1; i++)
          printf("%.2f ", T[i]); 

     puts("");  
} 
int main(int argc, char *argv[]) {

int n1 = atoi(argv[1]);
int n2 = atoi(argv[2]);
int n3 = atoi(argv[3]);
int rank, size, i;

MPI_Init(NULL, NULL);
double t1 = MPI_Wtime();

MPI_Status status;

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);


int chunk = n2 / size;

float *A,*A1,*A2,*x,*x_local,*c1_local,*c2_local,*matmul_local1,*matmul_local2,*result_vector1,*result_vector2,*result_matris;
float *x1_local,*x2_local,*x1,*x2,*matmul_local3,*matmul_local4,*result_vector3,*result_vector4;



x1_local = create1DArray(chunk*n3/2);
x2_local=  create1DArray(chunk*n3/2);

c1_local = create1DArray(chunk*n1/2);
matmul_local1 = create1DArray((n1/2)*(n3/2));


c2_local = create1DArray(chunk*n1/2);
matmul_local2 = create1DArray((n1/2)*(n3/2));

matmul_local3 = create1DArray((n1/2)*(n3/2));
matmul_local4 = create1DArray((n1/2)*(n3/2));


if (rank == MASTER) {
     A = create1DArray(n1 * n2);
     fillArray(A, n1 * n2);
     x = create1DArray(n2*n3);
     fillArray(x,n2*n3);

     result_vector1 = create1DArray((n1/2)*(n3/2));
     result_vector2 = create1DArray((n1/2)*(n3/2));
     result_vector3 = create1DArray((n1/2)*(n3/2));
     result_vector4 = create1DArray((n1/2)*(n3/2));
     result_matris =  create1DArray(n1*n3);


A1=create1DArray(n1 * n2/2);
A2=create1DArray(n1 * n2/2);
x1=create1DArray(n2*n3/2);
x2=create1DArray(n2*n3/2);


for (int i = 0; i < n1*n2/2; i++){
	A1[i] =A[i];
	A2[i] =A[(n1*n2/2)+i];
}


int a=0;
for(int z=0; z < n2; z++){
	float *b=&x[z*n2];
	for(int j=0; j < n3/2; j++){
	x1[a]=b[j];
	x2[a]=b[j+(n3/2)];
	a++;
}
}

//x1=change(x1,n2,n3/2);
//x2=change(x2,n2,n3/2);

	
}

MPI_Datatype colType, newColType;
int blocklength = chunk, stride = n2, count = n1/2;
MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &colType);
MPI_Type_create_resized(colType, 0, chunk*sizeof(float), &newColType);
MPI_Type_commit(&colType);
MPI_Type_commit(&newColType);

MPI_Datatype rowType;
MPI_Type_contiguous(n3/2, MPI_FLOAT, &rowType);
MPI_Type_commit(&rowType);


double t3 = MPI_Wtime();

MPI_Scatter(A1, 1, newColType, c1_local, chunk*n1/2, MPI_FLOAT, MASTER, MPI_COMM_WORLD); 
MPI_Scatter(A2, 1, newColType, c2_local, chunk*n1/2, MPI_FLOAT, MASTER, MPI_COMM_WORLD); 






MPI_Scatter(x1, chunk, rowType, x1_local, chunk*n3/2, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
MPI_Scatter(x2, chunk, rowType, x2_local, chunk*n3/2, MPI_FLOAT, MASTER, MPI_COMM_WORLD);  
MPI_Barrier(MPI_COMM_WORLD);

double t4 = MPI_Wtime();

x1_local=change(x1_local,chunk,n3/2);
x2_local=change(x2_local,chunk,n3/2);


matmul_local1=mat_vec_mult(c1_local, x1_local, n1/2, chunk,n3/2);  
matmul_local2=mat_vec_mult(c2_local, x1_local, n1/2, chunk,n3/2);

matmul_local3=mat_vec_mult(c1_local, x2_local, n1/2, chunk,n3/2);
matmul_local4=mat_vec_mult(c2_local, x2_local, n1/2, chunk,n3/2);

//if(rank==MASTER){
//printArray(c2_local,chunk*n1/2);
//printArray(x2_local,chunk*n3/2);
//printArray(matmul_local3,(n1/2)*(n3/2));
//printArray(matmul_local4,(n1/2)*(n3/2));

//}

//if(rank==1){
//printArray(c2_local,chunk*n1/2);
//printArray(x2_local,chunk*n3/2);
//printArray(matmul_local3,(n1/2)*(n3/2));
//printArray(matmul_local4,(n1/2)*(n3/2));
//}

//if(rank==2){
//printArray(c2_local,chunk*n1/2);
//printArray(x2_local,chunk*n3/2);
//printArray(matmul_local3,(n1/2)*(n3/2));
//printArray(matmul_local4,(n1/2)*(n3/2));
//}


//if(rank==3){
//printArray(c2_local,chunk*n1/2);
//printArray(x2_local,chunk*n3/2);
//printArray(matmul_local3,(n1/2)*(n3/2));
//printArray(matmul_local4,(n1/2)*(n3/2));
//}


double t5 = MPI_Wtime();

MPI_Barrier(MPI_COMM_WORLD);
MPI_Reduce(matmul_local1, result_vector1, (n1/2)*(n3/2), MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
MPI_Reduce(matmul_local2, result_vector2, (n1/2)*(n3/2), MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);

MPI_Barrier(MPI_COMM_WORLD);
MPI_Reduce(matmul_local3, result_vector3, (n1/2)*(n3/2), MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
MPI_Reduce(matmul_local4, result_vector4, (n1/2)*(n3/2), MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);

double t6 = MPI_Wtime();


//if(rank==MASTER){
//printArray(result_vector1,(n1/2)*(n3/2));
//printArray(result_vector2,(n1/2)*(n3/2));
//printArray(result_vector3,(n1/2)*(n3/2));
//printArray(result_vector4,(n1/2)*(n3/2));

//}


if(rank==MASTER){

int k=0;
for(int i=0; i<(n1/2);i++){

	float *first=&result_vector1[i*(n3/2)];
	float *third=&result_vector3[i*(n3/2)];
	
	for(int j=0; j<n3/2;j++){
          	result_matris[((n3/2)*i)+k]=first[j];
		result_matris[((n3/2)*i)+k+(n3/2)]=third[j];
		k++;
		
}
}

for(int i=n1/2; i<n1;i++){

	float *second=&result_vector2[(i-(n1/2))*(n3/2)];
	float *fourth=&result_vector4[(i-(n1/2))*(n3/2)];
	
	for(int j=0; j<n3/2;j++){
          	result_matris[((n3/2)*i)+k]=second[j];
		result_matris[((n3/2)*i)+k+(n3/2)]=fourth[j];
		k++;
		
}
}


//printArray(result_matris,n1*n3);
}
 

//MPI_Barrier(MPI_COMM_WORLD);
//if(rank==MASTER){
//printArray(result_matris,n1*n3);
//}



//if(rank==1){
//printArray(c1_local,n1/2);

//}

//if(rank == 1){
//printArray(x_local,chunk);

//}
double t2 = MPI_Wtime();

if(rank==MASTER){
 printf("Time elapsed TAll = %f sec.\n", t2 - t1);
 printf("Time elapsed TComm = %f sec.\n", ((t6-t5)+(t4-t3)));
printf("Time elapsed TComp = %f sec.\n", ((t2-t1)-((t6-t5)+(t4-t3))));
}





MPI_Finalize();

return 0;

}










#include<iostream>
#include<Eigen/Dense>
#include<cmath> 

using namespace std;
using namespace Eigen;

#define max_row 5
#define max_col 10

template<typename Derived, typename OtherDerived>
double abs_scalar(MatrixBase<Derived>& U, MatrixBase<OtherDerived>& V, long int rank){
	if( rank == 1){
		return 0;
	} else{
		double sum = 0;
		for (long int i=0; i < (rank-1);  i++ ){
			sum = sum + abs((U.col(i).transpose()*U.col(rank-1))(0,0)*(V.row(rank-1)*V.row(i).transpose())(0,0));
		}
		return(2*sum);
	}
}

template <typename Derived, typename OtherDerived, typename Dummy>
long int max_index(const MatrixBase<Derived>& M, const MatrixBase<OtherDerived>& N, const MatrixBase<Dummy>& D, long int vector_max_size){
	long int i,j,k,l;
	j=-1;
	if(N.size()==1){
		double max = abs(M(0));
		j = 0;
		for(i=1;i<vector_max_size;i++){
			if(abs(M(i))>max){
				max = abs(M(i));
				j=i;
			}
		}
		return j;
	}else{
		double max = 0;

		for(i=0;i<vector_max_size;i++){
			bool flag = true;
			for(k=0;k<N.size();k++){
				if((N(k)==i)){
					flag = false;
				}
			}
			for(l=0;l<D.size();l++){
				if(D(l)==i){
					flag = false;
				}
			}

			if(flag && (abs(M(i))>max || abs(M(i))==max)){
				max = abs(M(i));
				j=i;
			}
		}
		return j;
	}
}

int main(){

	float tol;
	cout<<"Enter tolerance: "<<endl;
	cin>>tol;

	MatrixXd Z(max_row, max_col);
	//Intializing Z
	cout<<"Enter the matrix entries:"<<endl;
	for(int x=0; x<max_row; x++){
		for(int y=0; y<max_col; y++){
			cout<<"Z("<<x<<","<<y<<"):"<<endl;
			cin>>Z(x,y);
			cout<<endl;
		}
	}

	MatrixXd Z_approx(max_row, max_col);
	MatrixXd R(max_row, max_col);
	
	long int k = 0;

	RowVectorXi I(k);
	VectorXi J(k);
	VectorXi D(k);
	VectorXi A(k);

	MatrixXd U(max_row, k);
	MatrixXd V(k, max_col);

	Z_approx = MatrixXd::Zero(max_row, max_col);
	double z = Z_approx.squaredNorm();
		
	//Randomly initialization of epsilon's value
	float epsilon = -10;

	while(abs(epsilon) >tol && k<max_row && k<max_col){
	
		cout<<endl<<"Beginning iteration: "<<(k+1)<<endl;

		I.conservativeResize(k+1);
		//Setting index as -2 to avoid random assignments due to resize
		I(k) = -2;
	
		if(J.size() == 0){
			I(k) = k;
		}else{
			I(k) = max_index(R.col(J(k-1)), I, D, max_row);
		}
		if(I(k) == -1){ 
			break;	
		}else{
			R.row(I(k)) = Z.row(I(k));
			for(int d=0;d<k;d++){ 
				R.row(I(k)) = R.row(I(k)) - U(I(k),d)*V.row(d);
			}
		}
		
		J.conservativeResize(k+1);
		//Setting index as -2 to avoid random assignments due to resize
		J(k) = -2;
		J(k) = max_index(R.row(I(k)), J, A, max_col);
		bool flag = false;
		if(J(k) == -1){
			break;
		}else{
			int j;
			if(abs(R(I(k),J(k)))<tol){
				while(abs(R(I(k),J(k)))<tol){
					j = D.size();
					D.conservativeResize(j+1);
					D(j)=I(k);
					I(k) = max_index(R.col(J(k-1)), I, D, max_row); 
					for(int d=0;d<k;d++){
						R.row(I(k)) = R.row(I(k)) - U(I(k),d)*V.row(d);
					}
					J(k) = max_index(R.row(I(k)), J, A, max_col);
					if(J(k) == -1 || I(k) == -1 || !(j<(max_row-k-1) && j<(max_col-k-1))){
						flag = true;
						break;
					}
				}
			}
			if(flag || J(k) == -1 || I(k) == -1){
				break;
			}

			V.conservativeResize(k+1,max_col);	
				V.row(k)= R.row(I(k))/R(I(k),J(k));
					R.col(J(k)) = Z.col(J(k));
					for(int d=0;d<k;d++){
						R.col(J(k)) = R.col(J(k)) - V(d,J(k))*U.col(d);
					}
			}
		
		U.conservativeResize(max_row,k+1);
		U.col(k) = R.col(J(k));
	
		z = z + V.row(k).squaredNorm()*U.col(k).squaredNorm() +abs_scalar(U,V,k+1);
	
		epsilon = (U.col(k).norm()*V.row(k).norm())/sqrt(z);
		cout<<endl<<"epsilon= "<<epsilon<<endl;
		cout<<endl<<"Ending iteration: "<<(k+1)<<endl;
	
		k++;
	}

	cout<<endl<<"U:"<<endl<<U<<endl<<"V:"<<endl<<V<<endl;
	cout<<endl<<"U*V"<<endl<<U*V<<endl;
	cout<<endl<<endl<<"rank of the matrix:"<<endl<<k<<endl;	
	cout<<endl<<"I:"<<endl<<I<<endl<<endl<<"J:"<<endl<<J<<endl;
	cout<<endl<<"Norm(Z-U*V)/Norm(Z)"<<endl<<(Z-U*V).norm()/Z.norm()<<endl;	
	cout<<endl<<"Norm(R)/Norm(Z)"<<endl<<R.norm()/Z.norm()<<endl;
	return 0;
}

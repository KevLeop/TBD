#include <iostream>
#include <string>
#include <cmath>
#include <limits>

using namespace std;

const float INF = numeric_limits<float>::infinity();
const int N = 8;

float tabla[N][8] = {{3.5,   2, INF, 4.5,  5, 1.5, 2.5,  2},
                     {  2, 3.5,   4, INF,  2, 3.5, INF,  3},
                     {  5,   1,   1,  3 ,  5,   1, INF,INF},
                     {  3,   4, 4.5, INF,  3, 4.5,   4,  2},
                     {INF,   4,   1,   4,INF, INF,   4,  1},
                     {INF, 4.5,   4,   5,  5, 4.5,   4,  4},
                     {  5,   2, INF,   3,  5,   4,   5,INF},
                     {  3, INF, INF,   5,  4, 2.5,   3,INF}};

string nombres[N] = {"Angelica", "Bill", "Chan", "Dan",
                     "Harley", "Jordyn", "Sam", "Veronica"};

int encontrarIndice(string nombre)
{
    for(int i = 0; i < N; ++i)
        if(nombre == nombres[i]) return i;

    return -1;
}

float manhattan(int idNombre1, int idNombre2)
{
    float suma = 0;
    for(int i = 0; i < N; ++i)
        if(tabla[idNombre1][i] != INF && tabla[idNombre2][i] != INF)
            suma += abs(tabla[idNombre1][i] - tabla[idNombre2][i]);

    return suma;
}

float euclidiana(int idNombre1, int idNombre2)
{
    float suma = 0;
    for(int i = 0; i < N; ++i)
        if(tabla[idNombre1][i] != INF && tabla[idNombre2][i] != INF)
            suma += pow(tabla[idNombre1][i] - tabla[idNombre2][i], 2.0);

    return sqrt(suma);
}

float minkowski(string nombre1, string nombre2, float p)
{
    int idNombre1 = encontrarIndice(nombre1);
    int idNombre2 = encontrarIndice(nombre2);

    if(idNombre1 == -1 || idNombre2 == -1) {
        cout << "Uno de los nombres es invalido" << endl;
        return 0;
    }

    float suma = 0, rpta = 0;

    for(int i = 0; i < N; ++i)
    {
        if(tabla[idNombre1][i] != INF && tabla[idNombre2][i] != INF)
            suma += pow(abs(tabla[idNombre1][i] - tabla[idNombre2][i]), p);
    }
    rpta = pow(suma, 1/p);
    return rpta;
}

float distanciaCoseno(int idNombre1, int idNombre2)
{
    float escalar = 0, normal1 = 0, normal2 = 0;
    for(int i = 0; i < N; ++i)
        if(tabla[idNombre1][i] != INF && tabla[idNombre2][i] != INF)
        {
            escalar += (tabla[idNombre1][i] * tabla[idNombre2][i]);
            normal1 += pow(tabla[idNombre1][i], 2.0);
            normal2 += pow(tabla[idNombre2][i], 2.0);
        }

    normal1 = sqrt(normal1);
    normal2 = sqrt(normal2);

    return escalar / (normal1 * normal2);
}

float aproximacionPearson(int idNombre1, int idNombre2)
{
    float suma1 = 0, suma2 = 0, suma3 = 0, suma4 = 0, suma5 = 0;
    float multi = 0, raiz1 = 0, raiz2 = 0, rpta = 0;
    for(int i = 1; i < N; ++i)
    {
        if(tabla[idNombre1][i] != INF && tabla[idNombre2][i] != INF)
        {
            suma1 += tabla[idNombre1][i] * tabla[idNombre2][i];
            suma2 += tabla[idNombre1][i];
            suma3 += tabla[idNombre2][i];
            suma4 += pow(tabla[idNombre1][i],2);
            suma5 += pow(tabla[idNombre2][i],2);
        }
    }
    multi += (suma2 * suma3)/N;
    raiz1 += suma4 - (pow(suma2,2))/N;
    raiz2 += suma5 - (pow(suma3,2))/N;
    raiz1 = sqrt(raiz1);
    raiz2 = sqrt(raiz2);
    return rpta = (suma1 - multi)/(raiz1*raiz2);
}

float similitud(string nombre1, string nombre2, string funcion)
{
    int idNombre1 = encontrarIndice(nombre1);
    int idNombre2 = encontrarIndice(nombre2);

    if(idNombre1 == -1 || idNombre2 == -1) {
        cout << "Uno de los nombres es invalido" << endl;
        return 0;
    }

    if(funcion == "manhattan") return manhattan(idNombre1, idNombre2);
    else if(funcion == "euclidiana") return euclidiana(idNombre1, idNombre2);
    else if(funcion == "aproximacionPearson") return aproximacionPearson(idNombre1, idNombre2);
    else return distanciaCoseno(idNombre1, idNombre2);
}

int main()
{
    float p = 2.0;
    cout << "La distancia manhattan es: " << similitud("Harley", "Veronica", "manhattan") << endl;
    cout << "La distancia euclidiana es: " << similitud("Harley", "Veronica", "euclidiana") << endl;
    cout << "La distancia Coseno es: " << similitud("Angelica", "Bill", "distanciaCoseno") << endl;

    cout << "La aproximacion de pearson es: " << similitud("Angelica", "Bill", "aproximacionPearson") << endl;
    cout << "La distancia minkowski con p="<< p <<" es: " << minkowski("Harley", "Veronica", p) << endl;

    return 0;
}

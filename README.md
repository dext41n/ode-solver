# Ordinary differential equation solver

Řeší diferenciální rovnice a vykreslí graf řešení. V souboru comparison
lze zhlédnout porovnání výsledků a chyb různých použitých metod. Všechny metody
jsou integrovány do solve_ode do funkce solve_ivp (initial value problem)

Metody: implicitní/explicitní eulerova metoda a Runge-Kutta metody - RK45
explicitní s adaptivním krokem a Radau implicitní metoda s pevným krokem.

Komplikovanější RK45 metoda používá i odhad chyby a podle něj vytváří další
krok. Obě RK metody jsou pátého řádu, konvergují o dost líp než euler.
Radau není implementovaná úplně super efektivně neboť obsahuje řešení nelineární
soustav 3n proměnných, což prý se dá udělat nějakou chytrou transformací
pro velké n efektivněji. Obě metody jsou implementovány pomocí Butcherovy
tabulky, což je elegantní způsob zápisu. Teorie je částečně i na wikipedii 
https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Adaptive_Runge%E2%80%93Kutta_methods

Program obsahuje i vlastní implementaci Newtonovi metody pro řešení nelineární
soustavy rovnic. Ta je obohacena ještě o LM metodu regularizace matice, což
pomáhá v některých případech divergence newtona. Navíc jsem implementoval verzi
s dampingem, který je schopen zkroti rychlé zvětšování kroku a divergenci
pro nějaké funkce, typicka arctg.

Pro uložení výsledků používám vlastní třídu Result, která jen obsahuje seznamy
t, x a derivací. Navíc tato třída umožňuje pomocí Hermitovské interpolace
zjistit přibližný výsledek v každém bodě. Lze použít i pro celé arraye a tím
získat něco jako dense output. Zvolil jsem Hermitovskou interpolaci, protože
využívá jen hodnot uzlů, a první derivace a pracuje po částech a je spojitá.
Normální kubický spline vyžaduje řešení další soustavy pro koeficienty, tak
jsem radši použil Hermitovu.

Obecně by implicitní metody měli mít lepší A-stabilitu řešení a měli by být
zejména lepší v řešení stiff problémů, což je ukázáno v comparison.

Comparison - Na prvním grafu lze vidět, jak se mění chyba při změně kroku metody
a rozdíl mezi přesností Eulerovy metody při hodně málem kroku a RK45 metody při
 kroku velikosti 1 na rovnici s jednoduýchým řešením.
Druhý příklad ukazuje, že řešení explicitním eulerem zvyšuje zachovávanou energii
a naopak řešení implicitním ji snižuje, zatímco obě Runge-Kutta metody si s tím
poradili obstojně a zachovávají energii.
Třetí graf ukazuje rozdíl mezi implicitní metodou při řešení stiff problémů. Obě
mají stejný krok a do třetice jsem tam dal pro porovnání ještě Radau.
Na čtvrtý příklad jsem přišel při testování eulera. Explictiní euler je pro tento
krok A-nestabilní, ale implicitní je stabilní. Explicitní exploduje bůhvíkam.
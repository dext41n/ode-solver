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
tabulky, což je elegantní způsob zápisu.

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

Obecně by implicitní metody měli mít lepší stabilitu řešení a měli by být
zejména lepší v řešení stiff problémů, což je ukázáno v comparison.

# Ordinary differential equation solver
Řeší diferenciální rovnice a vykreslí graf řešení. V souboru comparison
lze zhlédnout porovnání výsledků a chyb různých použitých metod. Všechny metody
jsou integrovány do solve_ode do funkce solve_ivp (initial value problem)

## Použití

```python
from solve_ode import solve_ivp

# x' = -x, x(0) = 1, na intervalu [0, 5]
sol = solve_ivp(lambda t, x: -x, 1, 0, 5, method="RK45")

x, t = sol.as_arrays()      # hodnoty v uzlech, které metoda skutečně použila
x_dense = sol(4.2)          # dense output -- hodnota v libovolném bodě intervalu
                             # (funguje pro skalár i pro pole časů)
```

## API -- `solve_ivp(f, x0, t0, t_end, method='RK45', graph=False, h=None, max_step=None, atol=1e-6, rtol=1e-3, adaptive=True, min_step=1e-12)`

Jednotné rozhraní pro všechny metody. Řeší `x' = f(t, x)`, `x(t0) = x0`
na `[t0, t_end]` a vrací objekt `Result`.

| Parametr | Popis | Metoda                                                    |
|---|---|-----------------------------------------------------------|
| `f` | pravá strana, callable `f(t, x)` | všechny                                                   |
| `x0` | počáteční podmínka (skalár nebo vektor) | všechny                                                   |
| `t0`, `t_end` | interval integrace | všechny                                                   |
| `method` | `"Euler"`, `"ImplicitEuler"`, `"RK45"`, `"Radau"` | --                                                        |
| `h` | délka kroku | `Euler`, `ImplicitEuler`, `Radau` (povinné)               |
| `max_step` | maximální krok | `RK45` (povinné, pokud `adaptive=False`, jinak jen strop) |
| `adaptive` | adaptivní krok ano/ne | jen `RK45`                                                |
| `atol`, `rtol` | tolerance pro adaptivní krok | jen `RK45` s `adaptive=True`                              |
| `min_step` | minimální povolený adaptivní krok | jen `RK45`                                                |
| `graph` | rovnou vykreslí `x(t)` | všechny                                                   |

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

## Známá omezení

- **Euler s pevným krokem nemusí trefit přesně `t_end`.** Počet kroků se počítá
  jako `round((t_end - t0) / h)`, takže pokud `h` nedělí délku intervalu beze
  zbytku, poslední uzel může být mírně za (nebo před) `t_end`. Implicitní varianty
  (`ImplicitEuler`, `Radau`) tohle řeší -- poslední krok zkrátí, aby skončil přesně v `t_end`.
- **Radau má velkou složitost.** Jeden krok řeší nelineární soustavu o `3*n`
  neznámých (kde `n` je rozměr `x`), protože se řeší všechny tři stage
  koeficienty najednou. Pro velké `n` je to citelně pomalejší než u Eulera.
- **Dense output (`sol(t_eval)`) ořezává body mimo `[t0, t_end]` beze
  slova.** Pokud do `sol(...)` pošlete pole s časy mimo interval integrace,
  ty se potichu zahodí a vrácené pole bude kratší než vstupní -- žádná
  extrapolace se nedělá. Při párování s jiným polem (např. s exaktním
  řešením pro výpočet chyby) je potřeba dát pozor, aby `t_eval` bylo vždy
  uvnitř `[t0, t_end]`.
- **Newtonova metoda v `ImplicitEuler`/`Radau` počítá Jacobián numericky**
  (centrální diference), pokud nedostane `jac`. To je robustní, ale
  pro velké soustavy nebo časté volání (mnoho kroků) to zbytečně stojí
  čas -- analytický/sparse Jacobián by byl výrazně rychlejší.

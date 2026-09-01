# Ordinary differential equation solver
Řeší diferenciální rovnice s počáteční podmíknou a vykreslí graf řešení. V souboru comparison
lze zhlédnout porovnání výsledků a chyb různých použitých metod. Všechny metody
jsou integrovány do funkce solve_ivp (initial value problem).

## Použití

```python
from solve_ode import solve_ivp

# x' = -x, x(0) = 1, na intervalu [0, 5]
sol = solve_ivp(lambda t, x: -x, 1, 0, 5, method="RK45")

x, t = sol.as_arrays()      # hodnoty v uzlech, které metoda skutečně použila
x_dense = sol(4.2)          # dense output -- hodnota v libovolném bodě intervalu
                             # (funguje pro skalár i pro pole časů)
```

## Solver -- `solve_ivp(f, x0, t0, t_end, method='RK45', graph=False, h=None, max_step=None, atol=1e-6, rtol=1e-3, adaptive=True, min_step=1e-12)`

Jednotné rozhraní pro všechny metody. Řeší `x' = f(t, x)`, `x(t0) = x0`
na `[t0, t_end]` a vrací objekt `Result`.

| Parametr | Popis                                             | Metoda                                                    |
|---|---------------------------------------------------|-----------------------------------------------------------|
| `f` | pravá strana, callable `f(t, x)`                  | všechny (povinné)                                         |
| `x0` | počáteční podmínka (skalár nebo vektor)           | všechny (povinné)                                         |
| `t0`, `t_end` | interval integrace                                | všechny (povinné)                                         |
| `method` | `"Euler"`, `"ImplicitEuler"`, `"RK45"`, `"Radau"` | --                                                        |
| `h` | délka kroku                                       | `Euler`, `ImplicitEuler`, `Radau` (povinné)               |
| `max_step` | maximální krok                                    | `RK45` (povinné, pokud `adaptive=False`, jinak jen strop) |
| `adaptive` | adaptivní krok True/False                         | jen `RK45`                                                |
| `atol`, `rtol` | tolerance pro adaptivní krok                      | jen `RK45` s `adaptive=True`                              |
| `min_step` | minimální povolený adaptivní krok                 | jen `RK45`                                                |
| `graph` | rovnou vykreslí `x(t)`                            | všechny (defauletně False)                                |

## Objekt `Results`

`solve_ivp` vrací objekt typu `Results`, ve kterém jsou uloženy výsledky, tedy trojice (x, t, dx).

### Vytvoření objektu a přidání výsledku

```python
result = Result(x=x0, t=t0, dx=dx0)
result.add(x_new, t_new, dx_new)
```
| Parametr | Popis             |
|---|-------------------|
| `x` | Hodnota řešení.   |
| `t` | Časový bod.       |
| `dx` | Hodnota derivace. |

### Přístup k uloženým datům

Hodnoty v krocích výpočtu jsou dostupné:

| Atribut | Význam |
|---|---|
| `result.x` | Seznam hodnot řešení v integračních uzlech. |
| `result.t` | Seznam časových bodů. |
| `result.dx` | Seznam derivací v jednotlivých uzlech. |

Pro převod hodnot `x` a `t` na pole NumPy slouží metoda `as_arrays()`:

```python
x, t = result.as_arrays()
```

### Dense output

Objekt `Results` lze volat jako funkci a tím lze získat interpolovaná hodnota
v jakémkoliv bodu uvnitř integračního intervalu. Jako argument lze použít rovnou
celé pole hodnot, což vrátí interpolované hodnoty ve všech bodech.

```python
import numpy as np

x_at_time = result(2.5)
t_array = np.linspace(0,10,1000)
x_dense = result(t_array)
```

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
 náročné na výpočet.


## Programátorská dokumentace

Program je členěn do více souborů, přičemž ten hlavní je solve_ode.py kde je i výsledná `solve_ivp`.
V solve_ivp je jen základní vykreslovací funkce `davinci`, jež umí vytvořit základní obrázky a pak ji volá,
pokud to uživatel zvolí `solve_ivp`. Dále jednotlivé metody jsou v samostatných souborech a `solve_ivp` je jen 
takovou hlavičkou, která je všechny volá. Pro potřeby jednotlivých metod, jsem dále implementoval Newtonovu v souboru newton.py.
Všechny výsledky jsou reprezentovány třídou `Result` kde se uchovává x,t,dx ale derivaci ukládám hlavně pro potřebu
interpolace při použití dense outputu.

### Newton

#### -- `newton(f, x0, jac = None, maxiter = 200, tol = 1e-9, reg = 1e-10):`
    Řeší soustavu f(x) = 0
    :param f: funkce callable
    :param x0: počáteční odhad
    :param jac: pokud ho máme spočtěný na papíře, formát jako matice
    :param maxiter: maximální počet itercí, default 200
    :param tol: tolerance na splnění rovnosti
    :param reg: počáteční regularizační faktor
    :return: Zpráva o konvergenci(True/False), počet potřebných iterací, kořen

Kde jakobián používá z funkce `jacobi` a případný damping, nebo regularizaci řeší funkce `damping`, `regularization`. Počítá inverz matice přes
řešení soustavy.

#### -- `jacobi(f,x,n):`
    """
    Aproximuje Jacobiho matici funkce f v bodě x centrálními diferencemi (lepší přesnost než jen na jednu stranu).
    :param f: callable, funkce Rn -> Rn
    :param x: bod, ve kterém se Jacobiho matice počítá
    :param n: dimenze Rn
    :return: aproximovaná Jacobiho matice, tvar (n, n)
    """

#### -- `regularization(J, fn, n, reg0 = 1e-10, max_tries = 10):`
    """
    Zkusí vyřešit soustavu J@delta = -fn pro regularizovanou matici (J + reg*I),
    pokud je J singulární. Regularizační faktor se při neúspěchu vždy desetkrát zvětší.
    :param J: (singulární) Jacobiho matice
    :param fn: hodnota f(x), pravá strana soustavy
    :param n: rozměr soustavy
    :param reg0: počáteční regularizační faktor
    :param max_tries: maximální počet pokusů se zvětšujícím se reg
    :return: řešení delta regularizované soustavy
    :raises numpy.linalg.LinAlgError: pokud soustava zůstane singulární i po max_tries pokusech
    """

#### -- `damping(f, x, delta, norm_old, max_tries = 20):`
    """
    Zkracuje krok (line search s půlením), dokud se nezlepší reziduum. Nemusí pomoct.
    :param f: callable, funkce soustavy f(x) = 0
    :param x: současný bod
    :param delta: navržený Newtonův krok
    :param norm_old: norma rezidua v současném bodě, se kterou se porovnává zlepšení
    :param max_tries: maximální počet půlení kroku
    :return: (nový bod x, f(nový bod) nebo None při neúspěchu, bool jestli se podařilo zlepšit)
    """

### euler_methods

#### -- `euler(f, x0, t0, t_end, h, implicit = False):`
    """
    Řeší differenciální rovnici eulerovou explicitní/implicitní metodou
    :param f: callable funkce
    :param x0: počáteční vektor
    :param t0: počáteční čas
    :param t_end: konečný čas
    :param h: délka kroku
    :param implicit: True použije implicitní, False explicitního eulera
    :return: objekt výsledků
    """
V případě volby implicitního eulera se spouští v každém kroku funkce:
#### --`implicit_step(f, x_prev, t, h):`
    """
    Udělá jeden krok implicitního (zpětného) Eulera: řeší x_new = x_prev + h*f(t+h, x_new)
    Newtonovou metodou. Pokud Newton nezkonverguje, zkusí to s poloviční délkou kroku.
    :param f: funkce pravé strany, callable f(t, x)
    :param x_prev: hodnota x na začátku kroku
    :param t: čas na začátku kroku
    :param h: požadovaná délka kroku
    :return: (nová hodnota x, nový čas t + použitý krok)
    :raises RuntimeError: pokud Newton nezkonverguje ani po opakovaném půlení kroku
    """
### rk_explicit
#### -- `rk45_explicit(f, x0, t0, t_end, max_step = None,  adaptive = True, atol = 1e-6, rtol = 1e-3, min_step = 1e-12):`
    """
    Řeší diferenciální rovnici x' = f(t,x) numericky metodou Runge-Kutta 45, s adaptivním krokem.
    :param f: funkce pravé strany, klidně soustava
    :param x0: počáteční podmínka
    :param t0: počáteční čas
    :param t_end: konec intervalu času řešení
    :param max_step: maximální krok metody, není nutný vyplňovat
    :param rtol: relativní tolerance
    :param atol: absolutní tolerance
    :param adaptive: True/False jestli chceš použít adaptivní krok
    :param min_step: min. povolený step pro adaptivní krok, asi není nutný skoro nikdy měnit
    :return: objekt Result
    """
Tato zastřešovací funkce postupně volá funkce:
#### --`rk45_params():`
    """
    Butcherova tabulka pro Dormand-Prince RK45.
    :return: (A, c, b_4, b_5) -- matice A, uzly c, váhy metody 4. a 5. řádu
    """
#### --`count_coeficients(f, x, t, h, c, A, k1):`
    """
    Spočítá koeficienty k1 až k7 (stage hodnoty) pro jeden krok RK45.
    Díky vlastnosti metody je k1 předaný jako k7 z předchozího kroku.
    :param f: funkce pravé strany, callable f(t, x)
    :param x: hodnota x na začátku kroku
    :param t: čas na začátku kroku
    :param h: délka kroku
    :param c: uzly z Butcherovy tabulky
    :param A: matice z Butcherovy tabulky
    :param k1: první stage koeficient
    :return: matice k s řádky k1 až k7, tvar (7, n)
    """
#### --`rk45_step(f, x, t, h, `1, c, A, b_4, b_5, adaptive):`
    """
    Spočítá adaptivní krok RK45 metody. Parametry korespondují s Butcherovou tabulkou na wikipedii a
    předchozíma funkcema.
    :return Adaptivní: x, chybu, k7, Jinak: x a k7
    """
#### -- `adaptive_step(error, atol, rtol, x, x_new, h, safety, max_step, fmin=0.1, fmax=10, p=4):`
    """
    funkce počítá adaptivní krok pro rk45
    :param max_step: maximální povolený krok
    :param error: chybový vektor počítaný ve funkci step
    :param atol: absolutní tolerance
    :param rtol: relativní tolerance
    :param x: současný x ve kterým jsme
    :param x_new: nově vznikklý x minulým krokem
    :param h: minulý použitý krok
    :param safety: pojistka aby to moc nerostlo
    :param fmin: kolikrát se nejvýše může krok zmenšit
    :param fmax: kolikrát se nejvýše může krok zvětšit
    :param p: řád té méně řádové metody
    :return: nový krok, normovanou chybu vůči atol a rtol
    """
#### -- `first_step(f, x0, t0, atol, rtol, p=4):`
    """
    Hrubý odhad délky prvního kroku podle počáteční hodnoty a derivace v t0
    :param f: funkce pravé strany, callable f(t, x)
    :param x0: počáteční podmínka
    :param t0: počáteční čas
    :param atol: absolutní tolerance
    :param rtol: relativní tolerance
    :param p: řád metody nižšího řádu (pro odhad)
    :return: odhad délky prvního kroku
    """
First_step vymyslí první krok, v těch params jsou jen schované parametry pro Butcherovu tabulku.
### rk_implicit
#### -- `radau(f, x0, t0, t_end, h):`
    """
    Řeší rovnici implicitní runge kutta metodou, Radau IIA
    :param f: callable
    :param x0: počáteční podmínka
    :param t0: počátenčí čas
    :param t_end: koncový čas
    :param h: délka kroku, tu je povinná
    :return: objekt Results
    """
Opět tohle je zastřešující funkce, která postupně volá:

#### -- `butcher_radau():`
    """
    Butcherova tabulka pro implicitní Runge-Kuttovu metodu Radau (3 stage, řád 5).
    :return: (A, c, b) -- matice A, uzly c a váhy b metody
    """
#### -- `count_coefs(K, f, x, t, h, A, c):`
    """
    Sestaví reziduální soustavu G(K) = 0 pro implicitní stage koeficienty K
    (3n neznámých), kterou pak řeší newton().
    :param K: zploštělý vektor stage koeficientů k1..k3, délka 3n
    :param f: funkce pravé strany, callable f(t, x)
    :param x: hodnota x na začátku kroku
    :param t: čas na začátku kroku
    :param h: délka kroku
    :param A: matice z Butcherovy tabulky
    :param c: uzly z Butcherovy tabulky
    :return: zploštělý vektor rezidua G, délka 3n
    """
#### -- `radau_step(f, x, t, h, A, c, b, K_prev=None):`
    """
    Spočítá jeden krok radau metody.
    :param f: callable funkce
    :param x: současný bod
    :param t: současný čas
    :param h: délka kroku
    :param A: matice A z Butcherovy tabulky
    :param c: vektor c z Butcherovy tabulky
    :param b: vektor b z Butcherovy tabulky
    :param K_prev: vektor K z minulého kroku, použitý jako odhad pro newtona
    :return: nový x, nový K, použitý krok
    """

### results

V tomhle souboru je vytvořená třída `Result` do které jednotlivé funkce ukládají výsledky. Vše je uloženo jako python list,
nepoužil jsem numpy arrray, protože předem nemusí být daná velikost a do listu se líp přidává.

#### -- `Result:`
    """
    Třída pro výsledky, pro lepší manipulaci a pak vykreslování.

    Uchovává posloupnost (t, x, dx) z numerické integrace a umožňuje
    dense output přes Hermitovu interpolaci voláním Result(t_eval).

    :ivar x: list hodnot stavu v jednotlivých uzlech
    :ivar t: list časů uzlů
    :ivar dx: list derivací f(t,x) v jednotlivých uzlech
    """

Třída má metody `add` a `as_arrays`.

#### -- `add(self, x_new, t_new, dx):`
        """
        Přidá jeden krok řešení, trojici (x, t, dx)
        :param x_new: nové x
        :param t_new: nové t
        :param dx: nová derivace
        """

#### -- `as_arrays(self):`
        """
        Převede uložené seznamy uzlů na numpy pole.
        :return: dvojice (x, t) jako numpy pole
        """

Hermitovu interpolaci vytváří soustava funkcí `search_num(array, x):`, `search(array, t_eval):` a `clip(a, start, end):`.
Všechny tyto funkce jen "připravují" data pro interpolování.
#### -- `search_num(array, x):`
    """
    Najde index prvního prvku v seřazeném poli, který je >= x (na principu binary search).
    :param array: seřazené 1D pole
    :param x: hledaná hodnota
    :return: index prvního prvku >= x
    """
#### -- `search(array, t_eval):`
    """
    Pro každý bod v t_eval najde index intervalu (uzlu vlevo), kam bod patří.
    :param array: seřazené pole uzlů (rostoucí časy)
    :param t_eval: pole časů, pro které hledáme příslušný interval
    :return: pole indexů levých krajních bodů intervalů, stejné délky jako t_eval
    """

#### -- `clip(a, start, end):`
    """
    Ořízne hodnoty pole do intervalu [start, end].
    :param a: pole hodnot
    :param start: dolní mez
    :param end: horní mez
    :return: nové pole se všemi hodnotami oříznutými do [start, end]
    """
Výslednou interpolaci provádí magické `__call__`
#### -- `__call__(self, t_eval):`
        """
        Dense output pomocí Hermitovy interpolace mezi uzly.

        Funguje jak pro jeden skalární čas, tak pro pole časů. Body mimo
        [t0, t_end] jsou oříznuty (žádná extrapolace) -- pokud
        nějaký bod z t_eval padne mimo interval integrace, vrácené pole
        bude kratší než vstupní t_eval.
        :param t_eval: skalár nebo pole časů, ve kterých chceme hodnotu řešení
        :return: interpolovaná hodnota x(t_eval); skalár/1D pole podle vstupu
        """
### comparison
Obashuje jen vykreslovací funkce a výpočet chyby pro tvorbu grafů pro porovnání řešení od různých metod.

## Závěr

Při psaní programu jsem narazil na problémy určitých metod, které pro některé rovnici nemusí vůbec
konvergovat, zatímco jiné metody to vyřeší. Dále jsem se docela pral s implementací newtona, protože
jsem potřeboval, aby byla co nejrobustnější. Stávalo se, že někdy divergoval, čemuž velmi pomohlo zavedení
dampingu. Ještě se mi stávalo, že poslední krok kvůli fpa vyšel neskutečně malý, což jsem musel řešit
použitím nějakého malého epsilonu. 
Jinak jak bylo řečeno výše, program není bůhvíjak efektivní, zejména implicitní metody, neboť používají
newtonovu metodu, kde je velmi náročné počítat jakobián a inverzi matice.

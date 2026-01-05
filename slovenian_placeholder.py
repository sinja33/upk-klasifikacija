# Placeholder slovenian dataset - 50 articles

import pandas as pd
import os

SLOVENIAN_ARTICLES = [
    # POLITIKA (10 articles)
    {'text': 'Vlada Republike Slovenije je danes na redni seji sprejela nov zakon o davčni reformi. Minister za finance je pojasnil, da bo reforma prinesla spremembe v dohodninski lestvici. Poslanci opozicije so napovedali kritično razpravo v parlamentu. Koalicija je zagotovila, da bodo spremembe pravične za vse državljane.', 'category': 'Politika'},
    {'text': 'Predsednik vlade se je danes sestal z evropskimi komisarji v Bruslju. Pogovarjali so se o zeleni tranziciji in digitalnih reformah. Slovenija bo prevzela pomembno vlogo pri koordinaciji projektov. Ministri so predstavili ambiciozne načrte za prihodnost.', 'category': 'Politika'},
    {'text': 'Državni zbor je včeraj potrdil proračun za prihodnje leto. Koalicijske stranke so podprle predlog, medtem ko je opozicija glasovala proti. Največji delež sredstev bo namenjen zdravstvu in šolstvu. Proračun predvideva tudi pomembne investicije v infrastrukturo.', 'category': 'Politika'},
    {'text': 'Minister za zunanje zadeve je obiskal sosednje države. Pogovori so se osredotočili na regionalno sodelovanje in skupne infrastrukturne projekte. Podpisanih je bilo več memorandumov o sodelovanju. Odnosi med državami se nadaljujejo v pozitivni smeri.', 'category': 'Politika'},
    {'text': 'Lokalne volitve bodo potekale prihodnji mesec. Kandidati so predstavili svoje programe za razvoj mest in občin. Volivci bodo odločali o županih in svetnikih. Pričakuje se visoka volilna udeležba v večjih mestih.', 'category': 'Politika'},
    {'text': 'Parlament je danes razpravljal o novi zakonodaji za varstvo okolja. Poslanci so poudarili pomen trajnostnega razvoja. Sprejetje zakona je načrtovano za konec meseca. Nevladne organizacije so pozdravile predlog.', 'category': 'Politika'},
    {'text': 'Ministrstvo za delo pripravlja reforme na področju pokojninskega sistema. Socialnim partnerjem so predstavili osnutek sprememb. Sindikati so napovedali dodatne pogovore. Vlada želi sistem narediti bolj vzdržen za prihodnje generacije.', 'category': 'Politika'},
    {'text': 'Predsednik države je sprejel veleposlanike evropskih držav. Pogovarjali so se o mednarodnih odnosih in sodelovanju. Slovenija bo okrepila diplomatske stike z vsemi partnerji. Srečanje je potekalo v prijateljskem vzdušju.', 'category': 'Politika'},
    {'text': 'Opozicijske stranke so včeraj predstavile alternativni predlog proračuna. Kritizirali so vladne prioritete in predlagali drugačno razporeditev sredstev. Koalicija je predlog zavrnila kot nerealen. Razprava v državnem zboru se nadaljuje.', 'category': 'Politika'},
    {'text': 'Vlada je sprejela strategijo digitalizacije javne uprave. Projekt predvideva posodobitev vseh državnih storitev. Državljani bodo upravne zadeve lahko urejali spletno. Investicija v projekt znaša 100 milijonov evrov v petih letih.', 'category': 'Politika'},
    
    # ŠPORT (10 articles)
    {'text': 'Slovenska košarkarska reprezentanca je včeraj zmagala proti Litvi s 85:78. Luka Dončić je dosegel 32 točk in 12 skokov. Tekma je potekala v Stožicah pred razprodano dvorano. Navijači so ustvarili neverjetno vzdušje in podprli svojo ekipo do zmage.', 'category': 'Šport'},
    {'text': 'NK Maribor je v derbiju premagal Olimpijo z 2:1. Gola sta zadela Zahović in Kronaveter. Trener je bil zadovoljen z igro ekipe v drugem polčasu. Z zmago je Maribor prevzel vodstvo na lestvici.', 'category': 'Šport'},
    {'text': 'Smučar Žan Kranjec je osvojil drugo mesto na veleslalomu v Kranjski Gori. Zmagal je Norvežan Henrik Kristoffersen. Kranjec je pokazal odlično vožnjo v obeh vožnjah. Publika je slovenskega smučarja navdušeno pozdravila.', 'category': 'Šport'},
    {'text': 'Primož Roglič se pripravlja na Tour de France. Kolesarski as trenira v Španiji in kaže odlično formo. Cilj je uvrstitev na stopničke. Slovenska ekipa ima letos močno sestavo.', 'category': 'Šport'},
    {'text': 'Rokometna reprezentanca je kvalificirala na svetovno prvenstvo. V odločilni tekmi so premagali Hrvaško s 30:28. Navijači so proslavili uspeh v Ljubljani. Selektor je pohvalil borbenost moštva.', 'category': 'Šport'},
    {'text': 'Tina Maze je bila imenovana za ambasadorko zimskih športov. Legendarna smučarka bo promoviral šport med mladimi. Program predvideva obiske šol in treningov. Maze je navdušena nad novo vlogo.', 'category': 'Šport'},
    {'text': 'Slovenski odbojkarji so v ligi narodov premagali Argentino. Zmaga s 3:1 je ekipo približala finalnemu turnirju. Trener je bil zadovoljen s predstavo. Naslednja tekma je proti Franciji.', 'category': 'Šport'},
    {'text': 'Atletinja Anita Horvat je postavila nov slovenski rekord v teku na 800 metrov. Njen čas je bil 1:58.32. Uspeh je dosegla na mitingu v Stockholmu. S tem si je zagotovila normo za svetovno prvenstvo.', 'category': 'Šport'},
    {'text': 'Hokejska reprezentanca se pripravlja na olimpijske kvalifikacije. Igralci so začeli s pripravami v Jesenicah. Selektor je poklical najboljše hokejiste iz tujine. Cilj je uvrstitev na zimske olimpijske igre.', 'category': 'Šport'},
    {'text': 'Plavalka Janja Šegel je zmagala na mednarodnem tekmovanju v Ljubljani. V disciplini 200 metrov prosto je dosegla najboljši letošnji čas. Trenerji so zadovoljni z napredkom. Šeglova cilja na olimpijske igre.', 'category': 'Šport'},
    
    # TEHNOLOGIJA (10 articles)
    {'text': 'Slovensko start-up podjetje je razvilo novo aplikacijo za učenje jezikov z umetno inteligenco. Aplikacija uporablja napredne algoritme za personalizirano učenje. Investitorji so pokazali veliko zanimanje. Podjetje načrtuje širitev na evropski trg.', 'category': 'Tehnologija'},
    {'text': 'Nov pametni telefon slovenskega proizvajalca je prišel na trg. Naprava ima odlične tehnične specifikacije in konkurenčno ceno. Prva serija je bila razprodana v dveh dneh. Podjetje načrtuje dodatno proizvodnjo.', 'category': 'Tehnologija'},
    {'text': 'Raziskovalci na Institutu Jožef Stefan so razvili inovativen polnilnik za električna vozila. Nova tehnologija omogoča hitrejše polnjenje baterij. Patent je že prijavljen. Proizvajalci avtomobilov so izrazili zanimanje za sodelovanje.', 'category': 'Tehnologija'},
    {'text': 'Slovenska IT podjetja so prejela evropska sredstva za razvoj облачnih storitev. Projekti se osredotočajo na kibernetsko varnost in digitalizacijo. Financiranje znaša 5 milijonov evrov. Podjetja bodo zaposlila dodatnih 200 programerjev.', 'category': 'Tehnologija'},
    {'text': 'Google je odprl razvojni center v Ljubljani. Podjetje bo zaposlilo 100 programerjev za delo na projektih umetne inteligence. To je pomemben mejnik za slovensko IT sceno. Odprtje centra bo spodbodilo razvoj tehnološkega sektorja.', 'category': 'Tehnologija'},
    {'text': 'Slovenski razvijalci so ustvarili platformo za varno izmenjavo podatkov med bolnišnicami. Sistem uporablja blockchain tehnologijo za zaščito zasebnosti. Pilotni projekt se začenja v Ljubljani. Ministrstvo za zdravje podpira projekt.', 'category': 'Tehnologija'},
    {'text': 'Nova verzija slovenskega operacijskega sistema za pametne hiše je na voljo. Posodobitev prinaša izboljšano varnost in nove funkcije. Uporabniki lahko upravljajo vse naprave iz ene aplikacije. Sistem je kompatibilen z vsemi glavnimi proizvajalci.', 'category': 'Tehnologija'},
    {'text': 'Raziskovalci so razvili revolucionarno tehnologijo za shranjevanje energije. Nove baterije imajo trikrat večjo kapaciteto. Projekt je financiran iz evropskih sredstev. Komercialna proizvodnja se bo začela čez dve leti.', 'category': 'Tehnologija'},
    {'text': 'Slovensko podjetje je predstavilo robota za pomoč starejšim. Robot lahko opravlja osnovne gospodinjske naloge in nudi družbo. Napredna umetna inteligenca omogoča naravno komunikacijo. Prvi roboti bodo na voljo prihodnje leto.', 'category': 'Tehnologija'},
    {'text': 'Novi superračunalnik na univerzi bo najzmogljivejši v regiji. Sistem bo uporabljen za kompleksne znanstvene simulacije. Investicija znaša 10 milijonov evrov. Dostop bodo imeli raziskovalci iz vseh fakultet.', 'category': 'Tehnologija'},
    
    # GOSPODARSTVO (10 articles)
    {'text': 'Borzni indeks SBI TOP je danes zrasel za 2 odstotka. Vlagatelji so optimistični glede gospodarske rasti. Delnice bank in zavarovalnic so najbolj pridobile. Analitiki napovedujejo nadaljnjo rast v naslednjih mesecih.', 'category': 'Gospodarstvo'},
    {'text': 'Izvoz slovenskega gospodarstva se je v lanskem letu povečal za 8 odstotkov. Največji izvozni partnerji so Nemčija, Italija in Avstrija. Avtomobilska industrija je vodilna panoga. Podjetja so optimistična glede letošnjega leta.', 'category': 'Gospodarstvo'},
    {'text': 'Centralna banka je objavila napoved gospodarske rasti za prihodnje leto. BDP naj bi zrasel za 3,5 odstotka. Inflacija naj bi se umirila na 2 odstotka. Gospodarska gibanja so stabilna in predvidljiva.', 'category': 'Gospodarstvo'},
    {'text': 'Novo poslovno središče bo zgrajeno v Ljubljani. Investicija je vredna 50 milijonov evrov. Projekt bo ustvaril 300 novih delovnih mest. Gradnja se bo začela prihodnji mesec.', 'category': 'Gospodarstvo'},
    {'text': 'Turizem v Sloveniji beleži rekordne številke. Število prenočitev se je povečalo za 15 odstotkov. Najbolj priljubljene destinacije so Ljubljana, Bled in obala. Turistična sezona je presegla vsa pričakovanja.', 'category': 'Gospodarstvo'},
    {'text': 'Slovensko farmacevtsko podjetje je pridobilo pomemben kontrakt s Kitajsko. Vrednost posla presega 20 milijonov evrov. Podjetje bo izvažalo zdravila za zdravljenje srčnih bolezni. To je največji izvozni posel v zgodovini podjetja.', 'category': 'Gospodarstvo'},
    {'text': 'Cene nepremičnin v Ljubljani so se v zadnjem letu dvignile za 12 odstotkov. Povpraševanje presega ponudbo, še posebej v centru mesta. Strokovnjaki svetujejo previdnost pri investiranju. Vlada pripravlja ukrepe za ohladitev trga.', 'category': 'Gospodarstvo'},
    {'text': 'Nov trgovski center v Mariboru je odprl svoja vrata. Investicija je ustvarila 400 novih delovnih mest. Center vključuje 80 trgovin in restavracij. Prvi vikend je obiskalo več kot 30.000 ljudi.', 'category': 'Gospodarstvo'},
    {'text': 'Slovenska podjetja so lani investirala rekordnih 800 milijonov evrov v raziskave in razvoj. To je 15 odstotkov več kot leto prej. Največje investicije so v farmaciji in IT sektorju. Vlada nadaljuje s subvencijami za inovacije.', 'category': 'Gospodarstvo'},
    {'text': 'Minimalna plača se bo januarja zvišala na 1.250 evrov. Sindikat so zadovoljni s povišanjem. Delodajalci opozarjajo na stroške za podjetja. Skupno bo od zvišanja imelo koristi 120.000 zaposlenih.', 'category': 'Gospodarstvo'},
    
    # ZNANOST (10 articles)
    {'text': 'Slovenski raziskovalci so odkrili novo zdravilo proti redki bolezni. Klinične študije so pokazale odlične rezultate. Zdravilo bo kmalu na voljo pacientom. Odkritje je rezultat petletnega raziskovalnega dela.', 'category': 'Znanost'},
    {'text': 'Univerza v Ljubljani je sodelovala pri projektu Evropske vesoljske agencije. Slovenski znanstveniki so prispevali k razvoju satelitskega sistema. Satelit bo lansiran prihodnje leto. Projekt raziskuje podnebne spremembe.', 'category': 'Znanost'},
    {'text': 'Novi teleskop na Rogli omogoča opazovanje vesolja v visoki ločljivosti. Astronomi so že odkrili več novih objektov. Observatorij je odprt tudi za javnost. Vsak teden potekajo vodeni ogledi.', 'category': 'Znanost'},
    {'text': 'Raziskava o podnebnih spremembah kaže na dvig temperature v Alpah. Slovenski znanstveniki so analizirali podatke iz zadnjih 50 let. Rezultati so zaskrbljujoči. Potrebni so nujni ukrepi za zmanjšanje emisij.', 'category': 'Znanost'},
    {'text': 'Medicinski fakulteti je uspelo gojiti človeško tkivo v laboratoriju. Preboj bo omogočil boljše zdravljenje poškodb. Študija je objavljena v prestižni reviji Nature. Raziskovalci so prejeli mednarodno priznanje.', 'category': 'Znanost'},
    {'text': 'Arheologi so v Ljubljani odkrili ostanke rimske naselbine. Najdba vključuje več stavb in uporabnih predmetov. Odkritje spreminja razumevanje zgodovine mesta. Izkopavanja se bodo nadaljevala naslednje leto.', 'category': 'Znanost'},
    {'text': 'Raziskovalci so razvili novo metodo za čiščenje podtalnice. Tehnologija uporablja posebne bakterije za razgradnjo onesnaževal. Pilotni projekt se izvaja v Mariboru. Rezultati presegajo pričakovanja.', 'category': 'Znanost'},
    {'text': 'Slovenski biologi so odkrili novo vrsto metulja v Triglavskem narodnem parku. Vrsta je endemična za Slovenijo. Odkritje poudarja pomen ohranjanja biotske raznovrstnosti. Park bo okrepil zaščitne ukrepe.', 'category': 'Znanost'},
    {'text': 'Študija slovenskih psihologov kaže pozitivne učinke meditacije na duševno zdravje. Raziskava je trajala dve leti in vključila 500 udeležencev. Rezultati so bili objavljeni v mednarodni reviji. Metoda bo vključena v terapevtske programe.', 'category': 'Znanost'},
    {'text': 'Fiziki so na Institut Jožef Stefan razvili novo metodo za merjenje gravitacijskih valov. Naprava je stokrat občutljivejša od obstoječih. Projekt sodeluje z mednarodnimi raziskovalnimi centri. Pričakujejo prve meritve v prihodnjem letu.', 'category': 'Znanost'},
]


def create_slovenian_dataset(output_dir='data'):
    
    # Create DataFrame
    df = pd.DataFrame(SLOVENIAN_ARTICLES)
    df['clean_text'] = df['text'].str.lower()
    
    # Create numeric targets
    categories = sorted(df['category'].unique())
    category_to_id = {cat: i for i, cat in enumerate(categories)}
    df['target'] = df['category'].map(category_to_id)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'slovenian_news_placeholder.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    return df, categories


if __name__ == "__main__":
    df, cats = create_slovenian_dataset()
    print(f"\n✓ Placeholder dataset created with {len(cats)} categories and 50 articles")

"""
Dummy database mobil.
Di production, ganti dengan koneksi ke PostgreSQL / SQLite via SQLAlchemy.
"""
from backend.models.schemas import CarSpec, CarColor

CAR_DB: dict[str, CarSpec] = {
    "outlander-sport-px": CarSpec(
        id="outlander-sport-px",
        brand="Mitsubishi",
        model="Outlander Sport",
        variant="PX",
        year=2014, 
        type="SUV",
        seats=5,
        price_otr_jakarta=150_000_000, # harga bekas
        engine_cc=1998,
        horsepower=150,
        fuel_consumption_kml=13.3, 
        wheel_size_inch=17,
        colors=[
            CarColor(name="White Pearl", hex="#F5F5F0"),
            CarColor(name="Cool Silver Metallic", hex="#85929E"),
            CarColor(name="Black Mica", hex="#2C2C2C"),
            CarColor(name="Titanium Gray Metallic", hex="#4A4A4A"),
            CarColor(name="Copper Metallic", hex="#B87333"),
            CarColor(name="Red Metallic", hex="#990000"),
        ],
        features=[
            "MIVEC Technology Engine",
            "INVECS-III CVT with 6-Step Sport Mode",
            "Magnesium Paddle Shift",
            "Panoramic Glass Roof Sensation",
            "Keyless Operating System (KOS) + Start Stop Engine Button",
            "7\" Touchscreen Entertainment System",
            "Super Wide Range HID Headlamps with Auto Leveling",
            "Dual SRS Airbags",
            "Anti-lock Braking System (ABS) + EBD + BA",
            "Brake Override System (Smart Pedal)",
            "RISE (Reinforced Impact Safety Evolution) Body",
        ],
        radar={"kenyamanan": 85, "performa": 78, "efisiensi": 70, "keamanan": 80, "kapasitas": 65},
    ),

    "xpander-cross": CarSpec(
        id="xpander-cross",
        brand="Mitsubishi",
        model="Xpander",
        variant="Cross",
        year=2024,
        type="SUV MPV",
        seats=7,
        price_otr_jakarta=362_000_000,
        engine_cc=1499,
        horsepower=105,
        fuel_consumption_kml=13.6,
        wheel_size_inch=17,
        colors=[
            CarColor(name="White Diamond", hex="#F0F0EC"),
            CarColor(name="Black", hex="#1A1A1A"),
            CarColor(name="Wine Red", hex="#8B0000"),
            CarColor(name="Bronze Gold", hex="#C0A020"),
            CarColor(name="Steel Blue", hex="#4A6280"),
        ],
        features=[
            "Adaptive Cruise Control",
            "Blind Spot Warning",
            "Rear Cross Traffic Alert",
            "Electric Parking Brake",
            '8" Touchscreen Infotainment',
            "6 Airbag",
        ],
        radar={"kenyamanan": 80, "performa": 78, "efisiensi": 72, "keamanan": 85, "kapasitas": 88},
    ),

    "destinator-ultimate-cvt": CarSpec(
        id="destinator-ultimate-cvt",
        brand="Mitsubishi",
        model="Destinator",
        variant="Ultimate CVT 1.5L Turbo",
        year=2025, # Mengikuti tahun sirkulasi dokumen saat ini
        type="SUV",
        seats=7, # Berdasarkan kapasitas tempat duduk (Seating Capacity)
        price_otr_jakarta=480_000_000, # Tidak tercantum di dalam brosur spesifikasi
        engine_cc=1499, # Berdasarkan kolom Displacement (cc)
        horsepower=163, # Berdasarkan kolom Maximum Power (120 kW (163 PS)/5000 rpm)
        fuel_consumption_kml=14.3, # Tidak tercantum di dalam brosur spesifikasi[cite: 2]
        wheel_size_inch=18, # Berdasarkan ukuran ban/velg "225/55 R18"[cite: 2]
        colors=[
            CarColor(name="Jet Black Mica", hex="#2C2C2C"),
            CarColor(name="Quartz White Pearl", hex="#F5F5F0"),
            CarColor(name="Blade Silver Metallic", hex="#85929E"),
            CarColor(name="Graphite Grey Metallic", hex="#4A4A4A"),
            CarColor(name="Lunar Blue", hex="#2E4057"),
        ],
        features=[
            "Diamond Sense Safety System (FCM, LCDN, RCTA, ACC, AHB, BSW, LCA)", # Paket keselamatan aktif varian Ultimate[cite: 2]
            "12.3-Inch Smartphone-Link Display Audio with Android Auto & Apple CarPlay", # Sistem infotainment utama[cite: 2]
            "Mitsubishi Connect (Automatic SOS, Roadside Assistance, Vehicle Tracking, etc.)", # Fitur konektivitas[cite: 2]
            "Panoramic Sunroof", # Atap kaca panorama[cite: 2]
            "Keyless Operation System with Engine Start/Stop Button", # Akses tanpa kunci[cite: 2]
            "Active Yaw Control (AYC)", # Fitur kontrol stabilitas menikung[cite: 2]
            "6 SRS Airbags (Driver, Passenger, Side, Curtain)", # Perlindungan kantong udara lengkap[cite: 2]
            "Electric Parking Brake with Autohold", # Rem parkir elektrik[cite: 2]
            "Dual Zone Automatic AC with Digital Display", # Pengatur suhu kabin otomatis[cite: 2]
            "Drive Mode Selector (Normal, Wet, Gravel, Tarmac, Mud)", # Pilihan mode berkendara[cite: 2]
        ],
        radar={"kenyamanan": 88, "performa": 85, "efisiensi": 78, "keamanan": 92, "kapasitas": 90},
    ),

    "eclipse-cross-ultimate": CarSpec(
        id="eclipse-cross-ultimate",
        brand="Mitsubishi",
        model="Eclipse Cross",
        variant="1.5L Turbo",
        year=2019, # Berdasarkan kode cetak brosur JUL/2019
        type="SUV",
        seats=5, # Berdasarkan Seating Capacity: 5
        price_otr_jakarta=240_000_000, # Tidak tercantum di dalam brosur spesifikasi
        engine_cc=1499, # Berdasarkan kolom Displacement (cc): 1499
        horsepower=150, # Berdasarkan Max Horse Power: 150 PS / 5500 rpm
        fuel_consumption_kml=9.3, # Tidak tercantum di dalam brosur spesifikasi
        wheel_size_inch=18, # Berdasarkan spesifikasi ban 225/55 R18 Alloy Wheel
        colors=[
            CarColor(name="Red Diamond", hex="#990000"),
            CarColor(name="Silky White", hex="#F5F5F0"),
            CarColor(name="Amethyst Black", hex="#2C2C2C"),
        ],
        features=[
            "1.5L Direct-Injection Turbocharged MIVEC Engine", # Teknologi mesin utama
            "INVECS-III CVT with 8-Speed Sport Mode & Paddle Shift", # Sistem transmisi sporty
            "Advanced Safety Features (FCM, ACC, AHB, BSW, LCA, RCTA, UMS, ASTC, HSA, AYC)", # Paket ADAS lengkap
            "7 SRS Airbags (Front, Side, Curtain, and Driver Knee Airbag)", # Perlindungan keselamatan pasif
            "Power Panoramic Sunroof", # Atap kaca ganda elektrik
            "Heads Up Display (HUD)", # Informasi kecepatan futuristik di dasbor
            "Touchpad Controller & Display Audio", # Kontrol sistem hiburan modern
            "Electric Parking Brake with Brake Auto Hold", # Rem parkir elektrik otomatis
            "Dual Zone Automatic Climate Control", # Pengatur suhu kabin otomatis dua zona
            "Leather Seats with Power Seat Adjuster & Heated Seat", # Kenyamanan kursi kulit elektrik berpemanas
        ],
        radar={"kenyamanan": 86, "performa": 82, "efisiensi": 76, "keamanan": 93, "kapasitas": 65},
    ),

    "triton-ultimate-4x4-at": CarSpec(
        id="triton-ultimate-4x4-at",
        brand="Mitsubishi",
        model="Triton",
        variant="Ultimate 4x4 AT",
        year=2024, # Berdasarkan kode cetak brosur JULI/2024
        type="Pickup Double Cabin",
        seats=5, # Berdasarkan konfigurasi Double Cabin 2 baris kursi
        price_otr_jakarta=635_000_000, # Tidak tercantum di dalam brosur spesifikasi
        engine_cc=2442, # Berdasarkan kolom Displacement (cc): 2442
        horsepower=184, # Berdasarkan Max Power Output: 184 PS / 3500 rpm
        fuel_consumption_kml=9.5, # Tidak tercantum di dalam brosur spesifikasi
        wheel_size_inch=18, # Berdasarkan ukuran ban 265/60 R18 Alloy Wheel
        colors=[
            CarColor(name="White Diamond", hex="#F5F5F0"),
            CarColor(name="Blade Silver Metallic", hex="#85929E"),
            CarColor(name="Graphite Gray Metallic", hex="#4A4A4A"),
            CarColor(name="Jet Black Mica", hex="#2C2C2C"),
        ],
        features=[
            "4N16 Engine Type DOHC 16 Valve Inline 4-Cylinder dengan VGT Turbo", # Teknologi mesin diesel terbaru
            "Super Select 4WD-II System dengan Drive Mode", # Sistem penggerak 4 roda mutakhir
            "ADAS Features (AHB, FCM, RCTA, LCA, BSW)", # Paket asisten keselamatan aktif berkendara
            "7 SRS Airbags (Driver, Passenger, Knee, Side & Curtain)", # Jumlah airbag lengkap untuk tipe Ultimate
            "8-Inch Audio Head Unit dengan Android Auto & Apple CarPlay", # Sistem entertainment multimedia
            "Dual Zone Auto Climate Control AC", # Pengatur suhu AC otomatis dua zona
            "Active Yaw Control (AYC) & Active Stability & Traction Control (ASTC)", # Kontrol stabilitas berkendara
            "Driver Power Seat with Driver Lumbar Support", # Kursi pengemudi elektrik dengan penyangga pinggang
            "Hill Start Assist (HSA)", # Penahan kendaraan di tanjakan
            "Wireless Charger & Auto Dimming Rearview Mirror", # Fitur kenyamanan modern di dalam kabin
        ],
        radar={"kenyamanan": 80, "performa": 90, "efisiensi": 74, "keamanan": 90, "kapasitas": 85},
    ),

    # ==========================================
    # 1. MITSUBISHI XFORCE (VARIAN TERTINGGI)
    # ==========================================
    "xforce-ultimate-diamond-sense": CarSpec(
            id="xforce-ultimate-diamond-sense",
            brand="Mitsubishi",
            model="Xforce",
            variant="Ultimate with Diamond Sense",
            year=2024, # Berdasarkan kode cetak dokumen "SZOZANTE" (Brosur Ref: 2024)
            type="SUV",
            seats=5, # SUV 5-Seater kompak[cite: 9]
            price_otr_jakarta=430_000_000, # Tidak tercantum di dalam brosur spesifikasi[cite: 9]
            engine_cc=1499, # Berdasarkan kolom Displacement (cc): 1499[cite: 9]
            horsepower=105, # Berdasarkan Max Horse Power: 77 (105 PS) / 6000 rpm[cite: 9]
            fuel_consumption_kml=15, # Tidak tercantum di dalam brosur spesifikasi[cite: 9]
            wheel_size_inch=18, # Berdasarkan Wheel & Tire: 225/50 R18 Two-Tone Alloy Wheel[cite: 9]
            colors=[
                CarColor(name="Quartz White Pearl", hex="#F5F5F0"),
                CarColor(name="Blade Silver Metallic", hex="#85929E"),
                CarColor(name="Graphite Gray Metallic", hex="#4A4A4A"),
                CarColor(name="Jet Black Mica", hex="#2C2C2C"),
                CarColor(name="Energetic Yellow", hex="#D4AC0D"),
                CarColor(name="Red Metallic", hex="#990000"),
            ],
            features=[
                "Diamond Sense Safety System (FCM, ACC, BSW, RCTA, AHB, LCDN)", # Fitur ADAS lengkap varian DS[cite: 9]
                "Dynamic Sound YAMAHA Premium (8 Speakers, 4 Sound Profiles)", # Audio premium kolaborasi Yamaha[cite: 9]
                "12.3 inch Smartphone-Link Display Audio & 8 inch Digital Driver Display", # Layar kembar futuristik[cite: 9]
                "Dual Zone Auto AC with nanoeX Technology", # Penyejuk udara anti virus & bakteri[cite: 9]
                "Drive Mode Selector (Normal, Wet, Gravel, Mud)", # Pilihan mode berkendara multi-medan[cite: 9]
                "Active Yaw Control (AYC) & Active Stability Control (ASC)", # Sistem kontrol stabilitas manuver[cite: 9]
                "Handsfree Power Liftgate with Kick Sensor", # Pintu bagasi elektrik dengan sensor kaki[cite: 9]
                "Synthetic Leather Seat with Anti-temperature Rise Function", # Jok kulit penolak panas[cite: 9]
                "Floor Center Console Box with Cooling Function", # Konsol tengah dengan pendingin minuman[cite: 9]
                "Wireless Smartphone Charger & Ambient Lighting", # Fitur interior premium modern[cite: 9]
            ],
            radar={"kenyamanan": 88, "performa": 75, "efisiensi": 80, "keamanan": 92, "kapasitas": 65},
        ),

    # ==========================================
    # 2. MITSUBISHI NEW PAJERO SPORT (VARIAN TERTINGGI)
    # ==========================================
    "pajero-sport-dakar-ultimate-4x4-at": CarSpec(
            id="pajero-sport-dakar-ultimate-4x4-at",
            brand="Mitsubishi",
            model="Pajero Sport",
            variant="Dakar Ultimate 4x4 AT",
            year=2024, # Berdasarkan kode cetak brosur JULI/2024
            type="SUV",
            seats=7, # Berdasarkan Seating Capacity: 7
            price_otr_jakarta=790_000_000, # Tidak tercantum di dalam brosur spesifikasi[cite: 8]
            engine_cc=2442, # Berdasarkan kapasitas mesin tipe Dakar (4N15): 2442 cc[cite: 8]
            horsepower=181, # Berdasarkan Maximum Power: 133 kW (181 PS) / 3500 rpm[cite: 8]
            fuel_consumption_kml=12, # Tidak tercantum di dalam brosur spesifikasi[cite: 8]
            wheel_size_inch=18, # Berdasarkan Wheel & Tire: Two-tone Alloy Wheel 265/60 R18[cite: 8]
            colors=[
                CarColor(name="Quartz White Pearl", hex="#F5F5F0"),
                CarColor(name="Blade Silver Metallic", hex="#85929E"),
                CarColor(name="Graphite Gray Metallic", hex="#4A4A4A"),
                CarColor(name="Jet Black Mica", hex="#2C2C2C"),
            ],
            features=[
                "2.4L MIVEC Turbocharged Intercooled Diesel Engine (4N15 Euro 4)", # Mesin diesel bertenaga besar[cite: 8]
                "Super Select 4WD-II with Off-Road Mode (Gravel, Mud/Snow, Sand, Rock)", # Sistem traksi 4 roda legendaris[cite: 8]
                "8-speed Automatic Transmission with Paddle Shift", # Transmisi otomatis 8 percepatan responsif[cite: 8]
                "Advanced Safety Technology (FCM, ACC, BSW, LCA, RCTA, UMS)", # Fitur radar keselamatan aktif[cite: 8]
                "7 SRS Airbags & RISE Body Construction", # Perlindungan keselamatan pasif maksimal[cite: 8]
                "8 Inch Digital Driver Display & 8 Inch Touch Screen Display Audio", # Panel instrumen digital canggih[cite: 8]
                "Handsfree Power Liftgate with Kick Sensors", # Pintu bagasi otomatis via sensor tendangan[cite: 8]
                "Synthetic Leather Seat with Heat Guard (Black & Burgundy Two-Tone)", # Interior mewah penahan panas[cite: 8]
                "Dual Zone Auto AC with nanoeX & Power Sunroof", # Kenyamanan sirkulasi kabin kelas atas[cite: 8]
                "Hill Descent Control (HDC) & Hill Start Assist (HSA)", # Pengontrol kestabilan di medan turunan/tanjakan[cite: 8]
            ],
            radar={"kenyamanan": 90, "performa": 92, "efisiensi": 74, "keamanan": 94, "kapasitas": 90},
        ),

    # ==========================================
    # 3. MITSUBISHI NEW XPANDER (VARIAN TERTINGGI)
    # ==========================================
    "xpander-ultimate-cvt": CarSpec(
            id="xpander-ultimate-cvt",
            brand="Mitsubishi",
            model="Xpander",
            variant="Ultimate CVT",
            year=2025, # Berdasarkan kode cetak brosur MEI/2025
            type="MPV",
            seats=7, # Berdasarkan Seating Capacity: 7
            price_otr_jakarta=345_000_000, # Tidak tercantum di dalam brosur spesifikasi[cite: 10]
            engine_cc=1499, # Berdasarkan kolom Displacement (cc): 1499[cite: 10]
            horsepower=105, # Berdasarkan Maximum Power: 77 kW (105 PS) / 6000 rpm[cite: 10]
            fuel_consumption_kml=15, # Tidak tercantum di dalam brosur spesifikasi[cite: 10]
            wheel_size_inch=17, # Berdasarkan Wheel & Tire: Two-tone Alloy Wheel 205/55 R17[cite: 10]
            colors=[
                CarColor(name="Quartz White Pearl", hex="#F5F5F0"),
                CarColor(name="Blade Silver Metallic", hex="#85929E"),
                CarColor(name="Graphite Gray Metallic", hex="#4A4A4A"),
                CarColor(name="Jet Black Mica", hex="#2C2C2C"),
                CarColor(name="Red Metallic", hex="#990000"),
            ],
            features=[
                "1.5L MIVEC DOHC 16-Valve Engine (Euro 4 Standard)", # Mesin MIVEC hemat bensin & halus[cite: 10]
                "Smooth Transmission CVT (Continuously Variable Transmission)", # Transmisi halus minim getaran[cite: 10]
                "New Active Yaw Control (AYC) & Active Stability Control (ASC)", # Fitur kontrol manuver tikungan dan kestabilan[cite: 10]
                "New 6 SRS Airbags", # Peningkatan proteksi kantong udara kabin[cite: 10]
                "New 10 Inch Audio Touchscreen dengan Konektivitas Pintar", # Head unit ukuran masif terbaru[cite: 10]
                "New 8 Inch Digital Driver Display & New 3-Spoke Steering Wheel", # Dasbor digital modern layaknya kelas atas[cite: 10]
                "Electric Parking Brake with Brake Auto Hold", # Rem parkir elektrik otomatis ringkas[cite: 10]
                "Multi Around Monitor & Rear View Camera", # Kamera pemantau lingkungan parkir 360 derajat[cite: 10]
                "Wireless Smartphone Charger & Digital AC Control", # Pengisian daya nirkabel modern[cite: 10]
                "Keyless Operating System (KOS) with Push Start Button", # Akses masuk dan nyalakan mesin tanpa kunci[cite: 10]
            ],
            radar={"kenyamanan": 82, "performa": 68, "efisiensi": 84, "keamanan": 85, "kapasitas": 95},
        ),
}


def get_all_cars() -> list[CarSpec]:
    return list(CAR_DB.values())


def get_car_by_id(car_id: str) -> CarSpec | None:
    return CAR_DB.get(car_id)


def search_cars(query: str) -> list[CarSpec]:
    """Pencarian sederhana berdasarkan brand/model/type."""
    q = query.lower()
    return [
        car for car in CAR_DB.values()
        if q in car.brand.lower()
        or q in car.model.lower()
        or q in car.type.lower()
        or q in car.variant.lower()
    ]

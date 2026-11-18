"""E1-E3: external heat-reuse evidence table and its join to the fixed 1,216-record inventory.

The documentary evidence below was collected by targeted web search on 2026-11-07 and read by delegated reader agents
(quotes are as extracted by them; the authors should spot-check before citation). Case SELECTION was driven by documentary
evidence and inventory coverage, not by ranks: ranks were joined only after the case list and status classes were fixed
(some ranks were incidentally visible while locating records; see report section 2).

Status classes:  Operational | Under construction / committed | Announced / planned | Unverified
Match confidence: High (source and OSM record agree on operator/name and place; no competing record) |
                  Medium (agree on operator and place, but the building within a multi-building cluster is not identified) |
                  Ambiguous (several plausible records, source gives no site detail) | No match (no record in the 1,216)
"""
from e_lib import *

RC = "search+full-page read"      # source_check values
S = "search result only (page not read)"

CASES = [
 dict(case_id="E01", origin="initial development", project="Equinix PA10 - Olympic Aquatics Centre / Plaine Saulnier heat network", facility="Equinix PA10", country="FR",
      location="Saint-Denis (Paris region)", operator="Equinix", partner="Saint-Denis urban heat network (SMIREC / Plaine Commune Energie; operator named differently across sources)",
      status="Operational", status_date="2024-07-26", strength="Moderate",
      basis="Two July-2024 trade-press reports use present tense for heat to the Olympic Aquatics Centre pool; Equinix's own Jan-2022 release was future tense; no utility/post-Games delivery document was read.",
      quote="\"The Equinix PA10 data center is being used to heat the impressive Olympic Aquatic Centre in Paris.\" (DCK, 26 Jul 2024); \"the company is indirectly providing heat for the Olympic swimming pool\" (Techzine, 17 Jul 2024)",
      urls="https://datacenterknowledge.com/sustainability/paris-2024-excess-data-center-heat-used-to-warm-olympic-swimming-pools | https://www.techzine.eu/blogs/infrastructure/122398/inside-equinix-pa10-paris-datacenter-sets-green-example | https://investor.equinix.com/news-events/press-releases/detail/16/digital-business-boom-and-sustainability-driven-initiatives",
      src_types="trade press | trade press | operator press release", src_dates="2024-07-26 | 2024-07-17 | 2022-01-17", check=RC,
      ids=["FR_way_1121454054"], conf="High",
      reason="OSM way named 'Equinix PA10' (operator Equinix) at the Saint-Denis Equinix cluster; unique PA10 record. Caveat: PA9X (FR_way_1121454055, 77 m), PA3 (112 m) and PA2 (198 m) lie nearby (campus ambiguity, E5); no source says the other buildings share the heat link."),
 dict(case_id="E02", origin="initial development", project="Telia Helsinki Data Center (Pitajanmaki) - Helen district heating", facility="Telia Helsinki Data Center (Telia HDC)", country="FI",
      location="Pitajanmaki, Helsinki", operator="Telia Finland", partner="Helen Oy",
      status="Operational", status_date="2025-05-26", strength="Strong",
      basis="Utility news release states waste heat has been channelled into the Helsinki district-heating network since early 2023; expansion announced 2025. 2021 release was future tense (planned from June 2022).",
      quote="\"Since early 2023, Helen and Telia have been channeling waste heat from Telia's Pitajanmaki data center into the district heating network to warm Helsinki homes and properties.\"",
      urls="https://www.helen.fi/en/news/2025/helen-and-telias-collaboration-brings-even-more-data-center-heat-to-helsinki-residents | https://www.helen.fi/en/news/2021/konesalien-hukkalammosta-hiilineutraalia-kaukolampoa",
      src_types="utility | utility", src_dates="2025-05-26 | 2021-04-22", check=RC,
      ids=["FI_way_640329703"], conf="High",
      reason="OSM way named 'Telia HDC' (operator Telia Finland Oyj) at 60.216N 24.878E = Pitajanmaki; single Telia data-centre record in Helsinki."),
 dict(case_id="E03", origin="initial development", project="Meta Odense (Tietgenbyen) - Fjernvarme Fyn heat pump / energy centre", facility="Meta Odense data centre", country="DK",
      location="Odense", operator="Meta", partner="Fjernvarme Fyn",
      status="Operational", status_date="2019-12-31", strength="Moderate",
      basis="Consultant (Ramboll) project page: energy centre 'ready for operation in late 2019', surplus heat redistributed to Odense district heating; undated page, completion 2019; Meta's 2018 post is pre-opening (future tense).",
      quote="\"A Ramboll-designed energy centre is redistributing surplus heat from Meta's data centre in Odense, Denmark, to the city's district heating network.\"",
      urls="https://www.ramboll.com/projects/energy/meta-surplus-heat-to-district-heating | https://tech.facebook.com/engineering/2018/4/odense-dc",
      src_types="consultant project page | operator blog", src_dates="not stated (completion 2019) | 2018-04-26", check=RC,
      ids=["DK_way_648731568", "DK_way_940299764"], conf="High",
      reason="Two OSM ways both named 'Meta Odense Data center' (~370 m apart); sources describe one data-centre site, so the two records are one project (campus convention, E3). Building-level identity within the campus is not resolvable from the sources."),
 dict(case_id="E04", origin="initial development", project="NorthC (ex-NLDC) Aalsmeer - Energy Hub Aalsmeer", facility="NorthC Datacenter Aalsmeer", country="NL",
      location="Hornmeer district, Aalsmeer", operator="NorthC (formerly NLDC)", partner="ESA BV (pool/sports centre De Waterlelie), child centre Triade, Fertiplant; municipality of Aalsmeer",
      status="Operational", status_date="2020-06-25", strength="Moderate",
      basis="Construction21 (25 Jun 2020): exchange started September 2019 and three buildings use the residual heat; NorthC states present-tense heat to 'a children's center, a nursery, and a sports complex including a swimming pool' (undated). Customers named differ between sources; no sports hall named. The original listed this case as 'Not confirmed'.",
      quote="\"Three buildings in the district are now using residual heat from the nearby data centre of NorthC\"; \"The exchange of energy started in September of 2019.\" (Construction21, 2020-06-25)",
      urls="https://www.construction21.org/amp/articles/h/energy-hub-aalsmeer-smart-exchange-of-heat-and-cold.html | https://stimular.nl/praktijkvoorbeelden/warmtenet-datacenter-nldc-aalsmeer-bespaart-327-ton-co2 | https://www.northcdatacenters.com/en/blogs/sustainable-data-center-absolutely/",
      src_types="trade press platform | foundation page (republishes Aug-2019 item) | operator", src_dates="2020-06-25 | not stated (2019 source) | not stated", check=RC,
      ids=["NL_node_2765259404"], conf="High",
      reason="OSM node named 'NorthC Datacenter Aalsmeer'; NorthC's Oude Meer site is a separate record (NL_node_11386475042, ~3.8 km away) and is not the Aalsmeer facility."),
 dict(case_id="E05", origin="stage 04", project="Digital Realty (ex-Interxion) Vienna Floridsdorf - Klinik Floridsdorf (Wien Energie)", facility="Digital Realty VIE1 & VIE2", country="AT",
      location="Floridsdorf, Vienna", operator="Digital Realty (ex-Interxion)", partner="Wien Energie",
      status="Operational", status_date="2023-12-31", strength="Moderate",
      basis="City of Vienna energy-planning page (undated; 'Baujahr 2023') states heat has been used since autumn 2023 via heat pumps (3 MW installed) and a 150 m pipe to the clinic; 2022 press report was future tense.",
      quote="\"Seit Herbst 2023 macht Wien Energie sich diesen Umstand zunutze ...\" (since autumn 2023 Wien Energie uses the data centre's waste heat for Klinik Floridsdorf)",
      urls="https://www.wien.gv.at/umwelt/energie-projekt-klinik-floridsdorf | https://futurezone.at/digital-life/rechenzentrum-wien-energie-floridsdorf-klinik-waerme-abwaerme/401989061",
      src_types="municipal (undated) | press", src_dates="not stated (2023) | 2022-04-28", check=RC,
      ids=["AT_way_31990236"], conf="High",
      reason="Only Digital Realty record in Vienna ('Digital Realty VIE1 & VIE2', coordinates in Floridsdorf district). Neither source gives a street address, so the match rests on operator + district + uniqueness."),
 dict(case_id="E06", origin="stage 04", project="Google Hamina - Haminan Energia heat recovery", facility="Google Hamina data centre", country="FI",
      location="Hamina", operator="Google", partner="Haminan Energia (CapMan Infra)",
      status="Operational", status_date="2026-05-18", strength="Strong",
      basis="Issuer (CapMan Infra) story: heat recovery plant 'now supplying' district heat; first heat Q4 2025, inaugurated 12 May 2026. Google's May-2024 post was future tense. Operational status therefore post-dates the 2025-01-01 OSM snapshot.",
      quote="\"A new heat recovery pump investment enabled by CapMan Infra is now supplying emission-free district heating in Hamina\"; \"first heat supplied in Q4 2025\"; \"inauguration took place on 12 May 2026\"",
      urls="https://capman.com/?p=124710 | https://irei.com/news/capman-infras-portfolio-company-haminan-energia-advances-local-energy-transition-through-new-waste-heat-facility/ | https://blog.google/company-news/inside-google/around-the-globe/google-europe/our-first-offsite-heat-recovery-project-lands-in-finland/",
      src_types="issuer/owner of utility | trade press (paywalled) | operator", src_dates="2026-05-18 | 2026-05-18 | 2024-05-20", check=RC,
      ids=["FI_way_323686447", "FI_way_748868810", "FI_way_1339135383", "FI_relation_4510018"], conf="High",
      reason="Four Google Inc. records within ~760 m of each other at Hamina (60.53-60.54N 27.11-27.12E) = one campus; the sources do not name which building the heat pump serves, so the four records are treated as one project."),
 dict(case_id="E07", origin="stage 04", project="NorthC (ex-NLDC) Eindhoven 1 - High Tech Campus heat network", facility="NorthC Datacenter Eindhoven (High Tech Campus)", country="NL",
      location="High Tech Campus, Eindhoven", operator="NorthC (formerly NLDC)", partner="High Tech Campus Eindhoven",
      status="Operational", status_date="2018-11-12", strength="Moderate",
      basis="IO+ (Nov 2018): 'NLDC is now supplying 2MW of heat to the High Tech Campus'; campus page (undated) lists NorthC with 'heat reuse' without quantities. Older evidence; current volumes not verified.",
      quote="\"NLDC is now supplying 2MW of heat to the High Tech Campus.\" (IO+, 2018-11-12)",
      urls="https://ioplus.nl/archive/en/datacenters-as-heating-of-the-future/ | https://www.hightechcampus.com/?p=7620",
      src_types="trade press | campus operator page", src_dates="2018-11-12 | not stated", check=RC,
      ids=["NL_node_3918634315"], conf="High",
      reason="OSM node 'NorthC Datacenter Eindhoven' at 51.408N 5.459E (High Tech Campus). 'Eindhoven 2' (NL_node_11426879977) is a different site (Eindhoven-Acht) and is not matched."),
 dict(case_id="E08", origin="stage 04", project="Yandex Mantsala - Nivos district heating", facility="Yandex Mantsala data centre", country="FI",
      location="Mantsala", operator="Yandex", partner="Nivos (local utility)",
      status="Operational", status_date="2019-04-22", strength="Moderate",
      basis="Operator blog (Apr 2019): excess heat transferred to Nivos since the 2015 completion, 'half the town's heat supply'; EU case study (Mar 2023) gives a 4.0 MW heat pump installed 2015 but not current operation. Current (post-2022) operating status NOT verified.",
      quote="\"Currently, the data centre's excess heat is providing half the town's heat supply\" (Yandex blog, 2019-04-22)",
      urls="https://www.yandex.com/company/blog/earth-day-2019 | https://smart-cities-marketplace.ec.europa.eu/insights/solutions/datacentre-supplies-local-heating-mantsala-finland",
      src_types="operator blog | EU case study", src_dates="2019-04-22 | 2023-03-13", check=RC,
      ids=["FI_way_846760920", "FI_way_1246864464"], conf="High",
      reason="Two Yandex records ~80 m apart at Mantsala (60.629N 25.263E); one data-centre site, treated as one project."),
 dict(case_id="E09", origin="stage 04", project="AWS Tallaght - Tallaght District Heating Scheme (Heatworks)", facility="AWS data centre, Tallaght", country="IE",
      location="Tallaght, South Dublin", operator="Amazon Web Services", partner="Heatworks / South Dublin County Council (Fortum)",
      status="Operational", status_date="2023-04-06", strength="Strong",
      basis="Irish Times and RTE (6 Apr 2023): scheme 'has begun operating', heat from a local Amazon data centre to council buildings and TU Dublin Tallaght; Interreg Europe (25 Jun 2026): operational since 2022. No source gives the data-centre address.",
      quote="\"Ireland's first large-scale district heating system has begun operating in Tallaght, Co Dublin, using excess heat from a local data centre.\" (Irish Times, 2023-04-06)",
      urls="https://www.irishtimes.com/environment/climate-crisis/2023/04/06/excess-heat-from-dublin-data-centre-to-warm-local-buildings-in-first-for-ireland | https://www.rte.ie/news/regional/2023/0406/1375678-district-heating | https://www.interregeurope.eu/good-practices/waste-heat-recovery-from-data-centre-for-district-heating-supply-tallaght-district-heating-scheme | https://www.aboutamazon.eu/news/amazon-web-services/local-community-buildings-in-ireland-to-be-heated-by-amazon-data-centre",
      src_types="national press | national broadcaster | EU policy database | operator", src_dates="2023-04-06 | 2023-04-06 | 2026-06-25 | 2020-12-14", check=RC,
      ids=["IE_way_825144681", "IE_way_37682585", "IE_way_37596453", "IE_way_37596447", "IE_way_697470448"], conf="Ambiguous",
      reason="Five 'Amazon Web Services' records lie in the Tallaght area (0.9-1.8 km from the town centre); the sources say only 'Tallaght' and that the energy centre is 'next door' to the data centre. Excluded from the primary quantitative set; ranks of all five reported as a sensitivity range."),
 dict(case_id="E10", origin="stage 04", project="Digital Realty Stockholm Kista - Stockholm Exergi", facility="Digital Realty Stockholm Kista campus (STO1-STO5)", country="SE",
      location="Kista, Stockholm", operator="Digital Realty (ex-Interxion)", partner="Stockholm Exergi",
      status="Unverified", status_date="not dated", strength="Weak",
      basis="Only a third-party listing (Baxtel, undated) states heat delivery since 2015; the Stockholm Data Parks release (Nov 2019) describes new Interxion halls with heat reuse as a decision, not a delivery. No utility/operator statement of actual delivery was read.",
      quote="\"Waste heat has gone to Stockholm Exergi's district heating network since 2015, with Digital Realty-owned heat pumps in the newest building.\" (Baxtel listing, undated)",
      urls="https://baxtel.com/campus/digital-realty-stockholm-kista-campus | https://stockholmdataparks.com/2019/11/12/three-new-data-centers-with-heat-recovery-in-stockholm-data-parks/",
      src_types="third-party listing | utility-linked initiative release", src_dates="not stated | 2019-11-12", check=RC,
      ids=["SE_relation_6125296"], conf="High",
      reason="OSM relation 'Digital Realty Stockholm STO1-STO5' (59.422N 17.917E = Kista). Match is sound but the evidence class is Unverified, so the case is excluded from the operational set."),
 dict(case_id="E11", origin="stage 04", project="Microsoft Hoje-Taastrup - VEKS", facility="Microsoft data centre, Hoje-Taastrup", country="DK",
      location="Hoje-Taastrup", operator="Microsoft", partner="VEKS / HTF",
      status="Under construction / committed", status_date="2024-03-18", strength="Moderate",
      basis="Invest in Denmark case page (18 Mar 2024): data centre 'currently under construction'; VEKS and Microsoft 'anticipate the initial deliveries ... during the winter months of 2025-2026'. No later status read.",
      quote="\"The data centre, currently under construction, will generate enough heat to warm upwards of 6,000 local households.\"",
      urls="https://investindk.com/cases/microsoft-data-centre-set-to-provide-surplus-heating-for-6-000-danish-homes",
      src_types="government investment agency", src_dates="2024-03-18", check=RC,
      ids=[], conf="No match", reason="CHECK_MSFT_DK"),
 dict(case_id="E12", origin="stage 04", project="Apple Viborg - Viborg Varme", facility="Apple Viborg data centre", country="DK",
      location="Viborg (Foulum)", operator="Apple", partner="Viborg Varme",
      status="Announced / planned", status_date="2024-09-30", strength="Moderate",
      basis="Danish trade press (30 Sep 2024): free surplus heat offered but not used because of a price cap (paywalled lead only). A later Computer Weekly report (not readable, HTTP 403) reportedly says the cap was removed; no delivery document found.",
      quote="\"a price cap prevents free surplus heat from Apple's data center from being utilized\" (DOI, 2024-09-30)",
      urls="https://www.doi.dk/en/vindkraft/artikel/viborgensere-maa-kigge-langt-efter-overskudsvarme-fra-apple-datacenter-1",
      src_types="trade press", src_dates="2024-09-30", check=RC,
      ids=["DK_way_481615983"], conf="High", reason="Only Apple record in Denmark (operator Apple, 56.483N 9.574E = Viborg/Foulum)."),
 dict(case_id="E13", origin="stage 04", project="Equinix AM4 Amsterdam Science Park - Diemen feasibility study", facility="Equinix AM4", country="NL",
      location="Amsterdam Science Park", operator="Equinix", partner="Municipality of Diemen",
      status="Announced / planned", status_date="2025-01-29", strength="Moderate",
      basis="Trade press (29 Jan 2025): letter of intent to study, within a year, integrating AM4 waste heat into a district-heating network. Study only.",
      quote="\"In het komende jaar zal de samenwerking de haalbaarheid onderzoeken van het integreren van restwarmte in een stadsverwarmingsnetwerk\" (the partnership will examine feasibility during the coming year)",
      urls="https://www.dutchitchannel.nl/news/591299/equinix-en-gemeente-diemen-tekenen-intentieverklaring",
      src_types="trade press", src_dates="2025-01-29", check=RC,
      ids=["NL_way_482532123"], conf="Medium",
      reason="OSM way named 'Equinix AM4' (operator tag missing) sits in the Science Park Equinix cluster with seven other Equinix records (four labelled AM3) within ~80 m; the AM4 label is unique but the cluster is a campus (E5)."),
 dict(case_id="E14", origin="stage 04", project="Digital Realty (ex-Interxion) AMS5 Schiphol-Rijk - Polderwarmte", facility="Digital Realty AMS5", country="NL",
      location="Schiphol-Rijk", operator="Digital Realty (ex-Interxion)", partner="Polderwarmte",
      status="Announced / planned", status_date="2021-10-18", strength="Weak",
      basis="Letter of intent reported by trade press (18 Oct 2021, via search result; page not read). Delivery not confirmed.",
      quote="(search-result snippet only) letter of intent with Polderwarmte for AMS5 waste heat to Schiphol-Rijk business parks",
      urls="https://dutchitchannel.nl/683447/interxion-levert-restwarmte-voor-polderwarmte-bedrijventerreinen-schiphol-rijk.html | https://datacenterdynamics.com/en/news/interxion-shares-waste-heat-with-business-parks-in-schiphol-amsterdam",
      src_types="trade press | trade press", src_dates="2021-10-18 | not stated", check=S,
      ids=[], conf="No match", reason="CHECK_AMS5"),
 dict(case_id="E15", origin="stage 04", project="Green Dielsdorf - local district heating", facility="Green data centre, Dielsdorf", country="CH",
      location="Dielsdorf ZH", operator="Green.ch", partner="local district-heating network (operator not named)",
      status="Unverified", status_date="2022-11-05", strength="Weak",
      basis="SRF (5 Nov 2022): data centre 'ready from January to supply waste heat ... as soon as the district heating network is in operation' (conditional); the same article also uses present tense. Operation not confirmed.",
      quote="\"... ab Januar bereit, Abwarme zu liefern ... sobald das Fernwarmenetz in Betrieb ist.\" (ready from January to supply waste heat once the network is operating)",
      urls="https://www.srf.ch/news/schweiz/datencenter-als-stromlieferant-in-dielsdorf-heizt-ein-stromfresser-3500-wohnungen-und-bueros",
      src_types="public broadcaster", src_dates="2022-11-05", check=RC,
      ids=[], conf="No match", reason="CHECK_DIELSDORF"),
]


def build():
    sc = load_scores().set_index("facility_id")
    fs = pd.read_csv(PC / "results" / "c5_facility_sensitivity_equal.csv").set_index("facility_id")
    ms = load_multiscale()
    ids_all = set(sc.index)
    # --- documentation of "no record" cases, computed from the inventory (no outside data)
    chk = {}
    chk["CHECK_MSFT_DK"] = "no record with operator/name 'Microsoft' in DK (%d DK records)" % int((sc.country == "DK").sum()) \
        if not sc[(sc.country == "DK")].apply(lambda r: "microsoft" in f"{r['name']} {r['operator']}".lower(), axis=1).any() else "UNEXPECTED MATCH"
    chk["CHECK_AMS5"] = "no record labelled AMS5 (Digital Realty Amsterdam records present: AMS3, AMS6, AMS7, AMS8); not matched" \
        if not sc.name.fillna("").str.contains("AMS5|AMS 5").any() else "UNEXPECTED MATCH"
    d = np.hypot((sc.lat - 47.4827) * 111320, (sc.lon - 8.4575) * 111320 * np.cos(np.radians(47.4827)))
    chk["CHECK_DIELSDORF"] = "no inventory record within 5 km of Dielsdorf (nearest %.1f km)" % (d.min() / 1000)
    rows, recs = [], []
    for c in CASES:
        for i in c["ids"]:
            assert i in ids_all, i
        reason = chk.get(c["reason"], c["reason"])
        r = {k: v for k, v in c.items() if k not in ("ids", "reason", "conf", "check")}
        r = {{"urls": "source_urls", "src_types": "source_types", "src_dates": "source_dates", "strength": "evidence_strength", "facility": "facility_name",
              "partner": "heat_reuse_partner", "basis": "evidence_basis", "quote": "key_statement", "status_date": "status_evidence_date"}.get(k, k): v for k, v in r.items()}
        r["status_date_exactness"] = "year-only: source undated, year-end used as a proxy" if c["case_id"] in ("E03", "E05") else "exact (as stated by source)"
        r.update(matched_ids=" | ".join(c["ids"]), n_matched_records=len(c["ids"]), match_confidence=c["conf"], match_reason=reason,
                 source_check=c["check"], retrieved=RETRIEVED)
        # primary-set rule: operational AND high-confidence match
        r["in_primary_operational_set"] = (c["status"] == "Operational") and c["conf"] == "High"
        rows.append(r)
        for i in c["ids"]:
            rec = dict(case_id=c["case_id"], facility_id=i, name=sc.loc[i, "name"], operator=sc.loc[i, "operator"], lat=sc.loc[i, "lat"], lon=sc.loc[i, "lon"],
                       score_B0QA=sc.loc[i, "echo_dc_score_equal"], rank_B0QA=int(sc.loc[i, "echo_dc_rank_equal"]))
            for s in SCALES:
                rec[f"rank_{s}"] = int(fs.loc[i, f"rank_{s}"])
                m = ms[(ms.facility_id == i) & (ms.scale == s)].iloc[0]
                rec[f"opp_{s}"], rec[f"feas_{s}"] = m.opportunity_score_equal, m.feasibility_score_equal
            assert rec["rank_2p5km"] == rec["rank_B0QA"]
            recs.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(recs)


if __name__ == "__main__":
    ev, rec = build()
    ev.to_csv(RESULTS / "e1_external_evidence_table.csv", index=False)
    rec.to_csv(RESULTS / "e1_case_record_ranks.csv", index=False)
    print(ev.groupby("status").size().to_string())
    print(ev[["case_id", "facility_name", "status", "match_confidence", "n_matched_records", "in_primary_operational_set"]].to_string(index=False))
    print(rec[["case_id", "facility_id", "rank_B0QA", "rank_1km", "rank_5km", "rank_10km"]].to_string(index=False))

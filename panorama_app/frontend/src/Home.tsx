import React, { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Lenis from "lenis";
import { ChevronDown, Upload } from "lucide-react";

gsap.registerPlugin(ScrollTrigger);

const STAGES = [
  {
    id: "stage1",
    num: "01",
    title: "Сегментация",
    img: "/stage-segmentation.jpg",
    color: "#12160f",
    text: "U-Net++ находит зоны оталькования на микрофотографии шлифа и строит вероятностную маску: для каждого пикселя понятно, тальк это или нет, и насколько модель в этом уверена.",
  },
  {
    id: "stage2",
    num: "02",
    title: "Классификация",
    img: "/stage-classification.jpg",
    color: "#151a12",
    text: "Классификатор относит образец к одному из трёх классов — рядовая, труднообогатимая или оталькованная руда — и показывает вероятность по каждому классу, а не только итоговую метку.",
  },
  {
    id: "stage3",
    num: "03",
    title: "Правка и экспорт",
    img: "/stage-export.jpg",
    color: "#181c11",
    text: "Геолог проверяет маску на панораме, поправляет её кистью там, где нужно, и выгружает один или несколько снимков архивом со статистикой и отчётом.",
  },
];

function Home({ onOpenTool }: { onOpenTool: () => void }) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const scanCardRef = useRef<HTMLDivElement | null>(null);
  const lenisRef = useRef<Lenis | null>(null);

  useEffect(() => {
    const scroller = rootRef.current?.closest(".siteContent") as HTMLElement | null;

    const lenis = new Lenis({
      wrapper: scroller ?? window,
      content: scroller ? rootRef.current ?? undefined : undefined,
      duration: 0.6,
      easing: (t: number) => 1 - (1 - t) * (1 - t),
      smoothWheel: true,
      wheelMultiplier: 1,
    });
    lenisRef.current = lenis;
    lenis.on("scroll", ScrollTrigger.update);
    let rafId = requestAnimationFrame(function raf(time: number) {
      lenis.raf(time);
      rafId = requestAnimationFrame(raf);
    });

    const ctx = gsap.context(() => {
      ScrollTrigger.defaults({ scroller: scroller ?? undefined });

      const navItems = gsap.utils.toArray<HTMLElement>(".stageNav li");

      ScrollTrigger.create({
        trigger: ".stageNav",
        start: "top top+=32",
        endTrigger: "#stage3",
        end: "bottom center",
        pin: true,
        pinSpacing: false,
      });

      gsap.utils.toArray<HTMLElement>(".stage").forEach((stage, i) => {
        ScrollTrigger.create({
          trigger: stage,
          start: "top center",
          end: "bottom center",
          toggleClass: { targets: navItems[i], className: "is-active" },
          onEnter: () => gsap.to(".stagesSection", { backgroundColor: STAGES[i].color, duration: 1, ease: "power1.inOut" }),
          onEnterBack: () => gsap.to(".stagesSection", { backgroundColor: STAGES[i].color, duration: 1, ease: "power1.inOut" }),
        });
        const img = stage.querySelector("img");
        if (img) {
          gsap.to(img, {
            yPercent: 14,
            ease: "none",
            scrollTrigger: { trigger: stage, start: "top bottom", end: "bottom top", scrub: 0.6 },
          });
        }
      });

      window.setTimeout(() => ScrollTrigger.refresh(), 300);
    }, rootRef);

    return () => {
      cancelAnimationFrame(rafId);
      lenis.destroy();
      lenisRef.current = null;
      ctx.revert();
    };
  }, []);

  const onHeroMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    gsap.to(scanCardRef.current, {
      x: x * 24,
      y: y * 18,
      rotateX: -y * 6,
      rotateY: x * 6,
      duration: 0.6,
      ease: "power3.out",
    });
  };
  const onHeroLeave = () => {
    gsap.to(scanCardRef.current, { x: 0, y: 0, rotateX: 0, rotateY: 0, duration: 0.8, ease: "power3.out" });
  };

  const smoothScrollTo = (target: Element | null) => {
    if (!target || !(target instanceof HTMLElement)) return;
    lenisRef.current?.scrollTo(target, { duration: 0.9, easing: (t: number) => 1 - (1 - t) * (1 - t) });
  };

  return (
    <div className="home" ref={rootRef}>
      <section className="hero" onMouseMove={onHeroMove} onMouseLeave={onHeroLeave}>
        <div className="landingBg" />
        <div className="landingScrim" />
        <div className="landingHero">
          <h1>
            АНАЛИЗ
            <br />
            ШЛИФОВ РУДЫ
          </h1>
          <p className="landingSub">
            Сегментация зон оталькования и классификация руды по микрофотографиям шлифов — с картой уверенности
            модели, ручной правкой маски кистью и экспортом отчёта.
          </p>
          <button className="landingCta" onClick={onOpenTool}>
            <Upload size={18} /> Загрузить изображение
          </button>
        </div>

        <div className="scanCard" ref={scanCardRef}>
          <div className="scanCardHeader">
            <span>SCAN_042 · РЕЗУЛЬТАТ</span>
            <span className="scanDot" />
          </div>
          <img src="/hero-scan.jpg" alt="Пример сегментации талька" />
        </div>

        <button
          className="scrollHint"
          onClick={() => smoothScrollTo(rootRef.current?.querySelector(".stagesSection") ?? null)}
          aria-label="Прокрутить вниз"
        >
          <ChevronDown size={22} />
        </button>
      </section>

      <section className="stagesSection">
        <h2 className="chapter">
          <span>001 —</span> Как это работает
        </h2>
        <div className="stagesLayout">
          <nav className="stageNav">
            <ul>
              {STAGES.map((s, i) => (
                <li key={s.id} className={i === 0 ? "is-active" : ""}>
                  <a
                    href={`#${s.id}`}
                    onClick={(e) => {
                      e.preventDefault();
                      smoothScrollTo(document.getElementById(s.id));
                    }}
                  >
                    {s.num} {s.title}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
          <div className="stagesList">
            {STAGES.map((s) => (
              <div id={s.id} key={s.id} className="stage">
                <div className="stageImage">
                  <img src={s.img} alt={s.title} />
                </div>
                <div className="stageText">
                  <p className="stageNum">{s.num}</p>
                  <h3>{s.title}</h3>
                  <p>{s.text}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="ctaBanner">
        <h2>Готовы попробовать на своём снимке?</h2>
        <button className="landingCta" onClick={onOpenTool}>
          <Upload size={18} /> Загрузить изображение
        </button>
      </section>
    </div>
  );
}

export default Home;

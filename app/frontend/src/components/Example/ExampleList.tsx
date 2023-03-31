import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        text: "Datos a tener en cuenta en la venta a corporativos sin CECOR?",
        value: "Datos a tener en cuenta en la venta a corporativos sin CECOR?"
    },
    { text: "¿Cuánto es el tiempo de entrega por delivery en AMBA?", value: "¿Cuánto es el tiempo de entrega por delivery en AMBA?" },
    { text: "Facturación de equipo a precio y/o forma de pago diferente a la acordada", value: "Facturación de equipo a precio y/o forma de pago diferente a la acordada" }
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};

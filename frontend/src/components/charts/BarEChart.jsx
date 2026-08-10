import React from 'react';
import ReactEChartsCore from 'echarts-for-react/esm/core.js';
import * as echarts from 'echarts/core';
import { BarChart } from 'echarts/charts';
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([BarChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer]);

export default function BarEChart({ option, height = 340 }) {
  return (
    <ReactEChartsCore echarts={echarts} option={option} style={{ height }} notMerge lazyUpdate />
  );
}

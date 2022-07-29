library("ggpubr")
library("data.table")
library("magrittr")
library("foreach")


auroc.dt <- foreach(temp.dataset=c("1", "291", "282", "273")) %do% {
    data.table(
        dataset=temp.dataset,
        CNN.AUROC=paste(sep="", "../../result/simulation/files.2/", temp.dataset, "_CNN_auc.txt") %>% {scan(file=., what=double())},
        Markonv.AUROC=paste(sep="", "../../result/simulation/files.2/", temp.dataset, "_MarkonvV_auc.txt") %>% {scan(file=., what=double())}
    ) %>% {.[, index:=.I]} %>% {.[, pct.of.Markonv.is.better.to.plot:=(sum(Markonv.AUROC > CNN.AUROC)/.N) %>% round(4) %>% {.*100}]}} %>%
    rbindlist %>%
    {.[, dataset.to.plot:=c("1"="1", "291"="2", "282"="3", "273"="4")[dataset]]} %>%
    {.[, title:=paste(sep="", dataset.to.plot, ":", pct.of.Markonv.is.better.to.plot, "%")]}

auroc.melt.dt <- melt(data=auroc.dt, id.vars=c("dataset", "dataset.to.plot", "index"), measure.vars=c("Markonv.AUROC", "CNN.AUROC"), variable.name="model", value.name="AUROC")[, model.to.plot := c("Markonv.AUROC"="Markonv-based", "CNN.AUROC"="Classical convolution-based")[model]]

{
    ggplot(auroc.melt.dt, aes(x=model.to.plot, y=AUROC, fill=model.to.plot)) -> .;
    . + geom_boxplot() -> .;
    . + scale_y_continuous(breaks=seq(0.5, 1, 0.1)) -> .;
    . + stat_compare_means(comparisons=list(c("Markonv-based", "Classical convolution-based")), method="wilcox.test", method.args=list(alternative="two.sided")) ->.;
    . + theme_pubr() -> .;
    . + facet_grid(~dataset.to.plot, scales="free_y") -> .;
    . + labs(x="", y="\n\nAUROC", fill="") ->.;       
    . + theme(text = element_text(size=20), axis.text.x=element_blank()) -> .;
    ggsave(filename="../../result/simulation/simulation.auroc.png", plot=., device="png", width=18, height=15, units="cm")
}

